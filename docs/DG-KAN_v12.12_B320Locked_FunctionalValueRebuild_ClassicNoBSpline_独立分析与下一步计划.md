# DG-KAN v12.12：B320 已锁定后的 Functional Value Rebuild 与 Classic No-BSpline 并行计划

> 生成日期：2026-05-23  
> 基于：v12.11 B320 FunctionalMechanism ClassicNoBSpline 执行复盘，以及 v12.10/v12.9 已落盘结果。  
> 公式格式：Typora 友好，只使用 `$...$` 与 `$$...$$`。  
> 硬约束：CE-only；no teacher；no distillation；no loss modification；no sampler/class weight；no dataset-name branch；no fake/proxy/CPU offload；B-spline 在本版继续冻结；Functional 未过 controls 前不得写成 official success。

---

# 0. 一句话结论

v12.11 的真实状态不是“没有进度”，而是项目阶段发生了变化：

$$
\boxed{\text{B320 base 已经基本锁定；functional update 仍然没有工作。}}
$$

这次最重要的结果不是某个 functional 候选变好，而是 **functional 旧评分体系被证伪**。P3v2 跑了 182 个候选，没有一个 Pareto pass；旧 P3 score 到 P4 gain 的相关性是负的，$P3\_score\_P4\_gain\_corr=-0.0071$；B320-bestFunctional 的 P4 short-run 是 $0/36$ pass。也就是说，当前 promotion score 不能预测真实训练收益。

因此下一步不能继续 F14/F15、lambda、branch damping、residual weight、cap/gain 这类小修。v12.12 的主目标必须是：

$$
\boxed{
\text{在已锁定的 B320 anchor 上，重建 functional value，}
\text{让 functional update 的局部诊断能够预测并转化为 P4/P5 训练收益。}
}
$$

经典基函数支线继续保留，但 B-spline 已冻结。本版 active families 是 Rational、Chebyshev、Wavelet、RBF/FastKAN、Fourier。它们不是主目标，也不能抢 B320 functional 主线；它们用于验证经典 basis 是否能形成另一类 efficient base 或提供 geometry insight。

---

# 1. 本轮结果的独立分析

## 1.1 B320 base 已经不是当前 blocker

v12.11 中 B320 anchor lock 的关键数字是：

```text
B320_anchor_strict_pass = 1
step_ratio_q90 = 0.39931987348441034
memory_ratio_q90 = 0.1295238095238095
mean_delta = 0.027669270833333332
worst_delta = -0.001953125
near_pass_rate = 1.0
AUC_step_ratio = 0.9336784156141542
AUC_time_ratio = 0.7424718100091173
ECE_delta = 0.01767905056476593
CouplingR2 = 0.2381078772319669
NoiseSignalLeak = 0.015539980493485928
LineC_nontearing_pass = 1
```

这说明 B320 已经同时满足：

```text
效率：step 和 memory 明显优于 MLP envelope；
任务：mean delta 为正，worst 只有 -0.00195，near pass = 1；
收敛：AUC-step 和 AUC-time 都低于 1；
几何：Line C nontearing pass。
```

所以现在继续把主精力放在 B320 的小修上是不合理的。B320 可以作为当前 anchor。后续 base 线只做 hardening 与防退化，不再以“找 base”为主目标。

## 1.2 这次真正证伪的是旧 functional promotion score

v12.11 的 P3v2 / P4 bridge 数据非常明确：

```text
P3v2_candidate_count = 182
P3v2_pareto_pass_count = 0
P3v2_mechanism_probe_rows = 153
P3_to_P4_bridge_rows = 180
P3_score_P4_gain_corr = -0.007065898088334216
B320_bestFunctional_P4_pass = 0 / 36
route = R2-P3ScoreNotPredictive
official_functional_success = 0
```

这不是一个普通失败。它说明：

$$
\boxed{\text{当前 P3 局部评分不是 noisy，而是基本不预测 P4。}}
$$

如果一个 promotion score 对 P4 gain 的相关性接近 0 甚至略负，那么继续围绕这个 score 改 lambda、改符号、改事件间隔，只会制造更多局部假阳性。

## 1.3 CouplingR2 已经不是难点，NoiseSignalLeak 与 Reservoir 才是难点

v12.11 的 constrained low-rank projector solve 很关键。它在 reservoir/noise/tail proxy 梯度组成的低秩子空间里解固定目标，结果显示：

```text
Constrained rows with CouplingR2_delta >= 0.02 = 9 / 9
Constrained rows with NoiseSignalLeak_delta <= -0.01 = 0 / 9
Constrained rows with RealSignalReservoirRatio_delta <= -0.01 = 0 / 9
Constrained CouplingR2_delta mean = 0.06423320274330117
Constrained NoiseSignalLeak_delta mean = 0.00036814436316490173
Constrained RealSignalReservoirRatio_delta mean = 0.00005267063776652018
Constrained holdout_loss_ratio mean = 1.000170792595637
Constrained CEp99_delta mean = 0.000324143303765191
```

这说明三件事：

```text
1. 提高 train-probe coupling 已经不难。
2. tail / holdout safety 也不是最主要瓶颈。
3. 真正难的是让真实 NoiseSignalLeak 与 RealSignalReservoirRatio 下降到 gate 需要的量级。
```

因此 functional 线的核心 blocker 应该重新定义为：

$$
\boxed{
\text{Line C proxy-gradient 可达的局部方向，}
\text{无法显著降低真实 P3v2 gate 中的 NoiseSignalLeak 与 RealSignalReservoirRatio。}
}
$$

这也解释了为什么 P4 失败原因里反复出现：

```text
CouplingR2_delta < 0.02
NoiseSignalLeak_delta > -0.01
control_gap_vs_best <= 0
AUC_time_delta > 0
```

旧方向要么提高 coupling 但动不了 noise/reservoir，要么 task-safe 但几何效应近似 no-op。

## 1.4 Functional 现在不是“差一点”，而是价值定义错了

v12.10 曾经出现 F14d-OrthDirectBranchDampingOnly 的强 P3 gap，但 P4 没有转化。v12.11 进一步证明，即使用新的 F-M1/F-M2/F-M3/F-M5 机制探针、projector-level release、5-event sequence、constrained low-rank solve，也没有 P3v2 Pareto pass。

所以我不会把当前状态解释成：

```text
functional 方向快成功了，只差阈值或 lambda。
```

更准确是：

```text
functional 方向还没有找到能产生真实 Line C Pareto 改善的机制。
旧 P3 分数、branch damping、orthogonal residual、projector low-rank solve 都不足以 official。
```

因此 v12.12 必须把 functional update 从“找一个好局部 delta”改成“重建可预测 value + 机制验证”。

## 1.5 经典基函数支线有进展，但没有 FamilyPass

B-spline 已冻结，本版不再 active。其他 family 的当前状态如下：

| Family | 当前状态 | 主要证据 | 当前 blocker |
|---|---|---|---|
| Rational | TaskBlocked | 多组候选 L3 pass + A4 pass，但 A5 fail | AUC/task trajectory，不是 kernel/expression |
| Chebyshev | TaskBlocked | B3s/B3v/B3w/B3x/B3ao 等 A4 pass，但 task 崩 | reservoir high / task trajectory |
| Fourier | ExpressionBlocked | L3 很快，但 A4 全 fail，K4 对 E6/E8 更差 | expression coverage |
| RBF/FastKAN | ExpressionBlocked | fused RBF L3 pass，input-cross/rank/orthoproj 仍 A4 fail | E6/E8 coverage 不够 |
| Wavelet | TaskBlocked | B5h baseline A4 fail，B5i/B5j/B5k 打开 A4，B5m/B5n/B5p/B5q task 改善但仍 fail | task trajectory plateau |
| B-spline | Frozen | KernelBlocked / rejected for this version | 不分配 active budget |

这条线的意义不是替代 B320，而是提供 basis-family insight。现在最值得吸收的结论是：

```text
1. Rational 已经不是 kernel blocker，也不是 A4 blocker；它是 A5 trajectory blocker。
2. Chebyshev/Wavelet 可以通过 sidecar 打开 A4，但 task 会崩，说明 expression 过关不等于可训练坐标健康。
3. RBF/Fourier 的效率很好，但 expression coverage 不够；继续堆 rank/order 没有解决 E6/E8。
```

---

# 2. 当前项目是否在正确道路上

答案是：**方向是对的，但下一步必须彻底换 functional 推进方式。**

正确之处：

```text
1. 没有降低 gate。
2. 没有把 diagnostic 写成 official success。
3. 没有按 dataset 调参。
4. B320 base 已经真正站住，终于有资格承载 functional official re-entry。
5. Line C 诊断已经能暴露 CouplingR2、NoiseSignalLeak、Reservoir 的差异，不再只看 accuracy。
```

需要停止的地方：

```text
1. 停止 F14/F15 residual / branch damping 小网格。
2. 停止用旧 P3 score 做 promotion。
3. 停止把 CouplingR2 单独提高当作 functional 成功。
4. 停止在 Rational 上继续降低 cap/gain。
5. 停止在 RBF 上继续堆 fixed sidecar rank。
6. 停止在 Fourier 上单纯增加 frequency order。
7. 停止把 classic family focused run 冒充 full family success。
```

当前项目已经从：

```text
有没有一个可用 PureKAN base？
```

推进到：

```text
在可用 B320 base 上，functional update 能不能提供独立机制收益？
```

这是研究阶段的实质推进。慢的原因不是没跑实验，而是之前长期没有合格 base，现在终于进入最难的因果机制验证阶段。

---

# 3. 离目标还差多远

## 3.1 距离 efficient PureKAN base

很近，甚至可以说当前 B320 已是 v12 系列的 base anchor。仍需做的是：

```text
1. 复验 B320 under wider protocol；
2. 锁定 candidate id、implementation path、profile path；
3. 防止后续 functional / family runner 混淆 B320/B321/B314 字段。
```

## 3.2 距离 functional update success

还远。不是没有信号，而是信号不能转化：

```text
P3v2 candidate = 182；
P3v2 Pareto pass = 0；
P4 pass = 0/36；
P3 score 到 P4 gain 相关性近似 0；
constrained solve 只能打开 CouplingR2，动不了 NoiseSignalLeak / Reservoir。
```

functional update 必须重新构造机制，不能继续围绕旧 score 或单步局部 delta。

## 3.3 距离 next-gen MLP claim

还不能 claim。最终 claim 至少需要：

```text
1. B320 base 在 external fair / broader protocol 下稳定；
2. functional update 在 B320 上独立改善 Line C 和 task trajectory；
3. functional update 打过 AdamWParallel、RandomMatchedNorm、SNR-only、NoOpMatchedOverhead、MLP analog controls；
4. classic family 支线有明确 status，不被无理由丢弃；
5. 速度/显存/收敛/表达/几何形成同一张 Pareto 证据表。
```

---

# 4. v12.12 总目标

v12.12 的总目标不是继续找 base，也不是继续扩大 functional candidate 数量，而是：

$$
\boxed{
\text{锁定 B320 anchor，重建 functional value，}
\text{让 functional diagnostic 能预测 P4/P5 训练收益。}
}
$$

具体分成四条并行线：

```text
Line A：B320 anchor freeze / hardening。
Line B：Functional value rebuild and mechanism validation。
Line C：Manifold-Channel Geometry Diagnostics 作为 official value source。
Line D：Classic No-BSpline family closure。
```

四条线的关系：

```text
Line A 提供稳定 base，不再频繁改 base；
Line C 定义 functional value，不再使用旧 P3 score；
Line B 验证 functional update 是否能因果改善 Line C 与 P4；
Line D 并行推进经典基函数，但不抢 B320 functional 主线。
```

---

# 5. 核心假设

## H-A：B320 已足够作为 functional official anchor

假设：

$$
\boxed{
\text{B320 在更宽协议下仍保持 strict efficient base status，}
\text{因此后续 functional 的失败不能再归因于 base 不合格。}
}
$$

成立标准：

```text
step_ratio_q90 <= 0.80 或至少 <= 1.00；
memory_ratio_q90 <= 0.30；
mean_delta >= 0；
worst_delta >= -0.003；
near_pass_rate = 1.0；
AUC_step_ratio <= 1.00；
AUC_time_ratio <= 1.00；
ECE_delta <= 0.02；
LineC_nontearing_pass = 1。
```

若不成立，Codex 先做：

```text
1. 检查 B320/B321/B314 字段混用；
2. 复跑 exact B320-only，不跑 selected candidate；
3. 分离 architecture timing 与 Line C diagnostic overhead；
4. 不改 dataset-specific 参数。
```

## H-B1：旧 P3 score 不可用，需要从 Line C Pareto 重建 value

假设：

$$
\boxed{
\text{functional promotion score 必须直接预测 P4/P5 的 Line C Pareto gain，}
\text{旧 P3 score 应废弃。}
}
$$

成立标准：

```text
新 value score 与 P4 strict gain 的 Spearman rho >= 0.30；
新 value score 的 top-k P4 pass rate 明显高于 random / old score；
新 score 不能只奖励 CouplingR2，必须同时约束 NoiseSignalLeak、Reservoir、AUC、CEp99。
```

若不成立，Codex 先做：

```text
1. 不继续加 functional 候选；
2. 扩充 P4-labeled replay dataset；
3. 把 P4 outcome 拆成 mechanism labels：coupling-only, noise-only, reservoir-only, task-harm, no-op；
4. 用 monotone score / Pareto rule，不训练黑箱 selector。
```

## H-B2：当前 local VJP / projector low-rank 子空间不足以降低真实 NoiseSignalLeak / Reservoir

假设：

$$
\boxed{
\text{单步局部 proxy-gradient 能打开 coupling，}
\text{但不能在真实 Line C 上达到 noise/reservoir 阈值。}
}
$$

成立标准：

```text
重复 constrained low-rank solve 后：
  CouplingR2_delta >= 0.02 的比例高；
  NoiseSignalLeak_delta <= -0.01 的比例仍低；
  RealSignalReservoirRatio_delta <= -0.01 的比例仍低。
```

如果 H-B2 继续成立，Codex 不要扩大局部 VJP 候选池，而要进入 H-B3/H-B4：结构性 reparameterization 或 multi-window optimizer-level mechanism。

## H-B3：functional update 需要从 additive parameter delta 改成 coordinate maintenance / optimizer-state transport

假设：

$$
\boxed{
\text{当前 additive delta 无法改变 noise/reservoir channel，}
\text{但 function-preserving coordinate maintenance 或 optimizer-state transport 可能改变后续训练轨迹。}
}
$$

成立标准：

```text
single event 后 task/logit drift 很小；
5-window 后 RealSignalReservoirRatio 或 NoiseSignalLeak 有累计下降；
P4 AUC_time 不恶化；
control_gap_vs_best >= 0.005 或 bootstrap CI lower >= 0；
不是 no-op：functional displacement norm 和 Line C delta 均超过下限。
```

若不成立，Codex 先做：

```text
1. 对每个 event 记录 optimizer moments before/after；
2. 比较 no-transport vs moment-transport；
3. 如果 transport 无效，停止该 reparameterization family；
4. 不把 coupling-only gain 写成 success。
```

## H-D：classic families 仍有可用 insight，但不是 v12.12 主胜负手

假设：

$$
\boxed{
\text{Rational/Chebyshev/Wavelet/RBF/Fourier 至少能提供 basis-specific blocker 证据，}
\text{但 v12.12 的主 claim 仍来自 B320 functional。}
}
$$

成立标准：

```text
每个 active family 输出明确 status；
每个 status 有 L3/A4/A5/LineC 证据；
不再运行 B-spline；
不把 focused MNIST seed0 结果 promoted 成 official FamilyPass。
```

---

# 6. Line A：B320 Anchor Freeze / Hardening

## 6.1 目标

确认 B320 是后续 functional official re-entry 的唯一 anchor，避免 B314/B321/B109 字段混用造成解释混乱。

## 6.2 实验设计

### A0：B320 exact reproduction

```text
candidate = B320 exact id
implementation = F3-triton-workspace-forward-delta-readout-proj-grad-learnableP
methods = B320, same-param MLP, same-step MLP, B314, B321 diagnostic
seeds = 0..9
datasets = MNIST, Fashion-MNIST, KMNIST
train_size = 1024
val_size = 512
test_size = 512
epochs = current v12.11 protocol
```

### A1：B320 protocol widening

只做 dataset-agnostic widening：

```text
train_size = 1024, 2048
batch_size = 128, 256
epoch budget = current, 2x diagnostic
profile_steps = fixed warm/measure split
```

禁止：

```text
dataset-specific LR；
dataset-specific gain；
class weight；
sampler；
post-hoc calibration。
```

## 6.3 必须记录的 CSV

`v1212_b320_anchor_hardening.csv`

```text
run_id
candidate_id
implementation_id
dataset
seed
train_size
batch_size
epoch_budget
step_ratio_q90
forward_ratio_q90
backward_ratio_q90
update_ratio_q90
memory_ratio_q90
mean_delta
worst_delta
near_pass
AUC_step_ratio
AUC_time_ratio
ECE_delta
NLL_delta
CEp99_delta
margin_p10_delta
LineC_nontearing_pass
CouplingR2
NoiseSignalLeak
RealSignalReservoirRatio
provenance_fake_count
proxy_row_count
cpu_offload_count
```

## 6.4 判断标准

B320 remains locked if:

$$
step\_ratio_{q90}\le 1.00,
$$

$$
memory\_ratio_{q90}\le 0.30,
$$

$$
mean\_delta\ge 0,
$$

$$
worst\_delta\ge -0.003,
$$

$$
AUC\_step\le 1.00,
$$

$$
AUC\_time\le 1.00,
$$

$$
LineC\_nontearing=1.
$$

## 6.5 可视化

```text
fig_A_b320_exact_scorecard.svg
fig_A_b320_seed_dataset_auc_matrix.svg
fig_A_b320_efficiency_memory_violin.svg
fig_A_b320_linec_vs_mlp.svg
fig_A_b320_protocol_widening_delta.svg
```

## 6.6 不满足条件时 Codex 先尝试

```text
Case A1: step/time 失败但 AUC-step 过
  先查 timing accounting、warmup、diagnostic hook，不改模型。

Case A2: AUC-step 失败
  先查 train trace、LineC trajectory、CEp99/margin，不做 dataset-specific repair。

Case A3: LineC fail
  强制 B320-only，不允许 selected candidate 自动换成 B321。

Case A4: B320 与 B321 混用
  终止 run，修 runner manifest 与 candidate_id propagation。
```

---

# 7. Line C：Manifold-Channel Geometry Diagnostics 升级为 value source

## 7.1 目标

Line C 不再只是附属诊断，而是 functional value 的唯一合法来源。v12.12 里 functional candidate 的 promotion 必须由 Line C Pareto 决定。

## 7.2 核心指标

对每个 update window $[t,t+\Delta]$，记录：

$$
\Delta U_B=f_{\theta_{t+\Delta}}(B)-f_{\theta_t}(B),
$$

$$
\Delta U_Q=f_{\theta_{t+\Delta}}(Q)-f_{\theta_t}(Q).
$$

ridge coupling：

$$
A_t=\arg\min_A \|\Delta U_Q-A\Delta U_B\|_F^2+\lambda\|A\|_F^2.
$$

$$
CouplingR^2=1-\frac{\|\Delta U_Q-A_t\Delta U_B\|_F^2}{\|\Delta U_Q\|_F^2+\epsilon}.
$$

Signal/reservoir sketch：

$$
\hat K_{BB}=\Phi\Phi^T,
$$

$$
\hat W_B=\sum_{\tau=t}^{t+\Delta}\hat K_{BB}(\tau).
$$

真实信号困在 reservoir：

$$
RealSignalReservoirRatio=\frac{\|P_{res}r_{real}\|^2}{\|r_{real}\|^2+\epsilon}.
$$

噪声泄漏进 signal channel：

$$
NoiseSignalLeak=\frac{\|P_{sig}r_{noise}\|^2}{\|r_{noise}\|^2+\epsilon}.
$$

## 7.3 新 value 规则

不要合成黑箱分数。使用 Pareto + hard veto：

Functional candidate $u$ 相对 base $b$ 的 Line C value 成立，当且仅当：

$$
\Delta CouplingR^2(u)\ge 0.02,
$$

$$
\Delta NoiseSignalLeak(u)\le -0.01,
$$

$$
\Delta RealSignalReservoirRatio(u)\le -0.01,
$$

$$
\Delta AUC_{time}(u)\le 0,
$$

$$
\Delta CEp99(u)\le 0.05,
$$

$$
control\_gap(u)\ge 0.005.
$$

同时必须不是 no-op：

$$
\|\Delta\theta_u\|/\|\Delta\theta_{AdamW}\|\ge 0.05,
$$

$$
|\Delta CouplingR^2|+|\Delta NoiseSignalLeak|+|\Delta RealSignalReservoirRatio|\ge 0.03.
$$

## 7.4 必须记录的 CSV

`v1212_linec_window_diagnostics.csv`

```text
run_id
candidate_id
update_type
control_id
dataset
seed
step
window_size
CouplingR2_before
CouplingR2_after
CouplingR2_delta
NoiseSignalLeak_before
NoiseSignalLeak_after
NoiseSignalLeak_delta
RealSignalReservoirRatio_before
RealSignalReservoirRatio_after
RealSignalReservoirRatio_delta
KernelDrift
signal_effective_rank
reservoir_fraction
top_eigen_share
CEp99_delta
ECE_delta
AUC_step_delta
AUC_time_delta
holdout_loss_ratio
update_norm_ratio
no_op_flag
pareto_pass
fail_reason
```

## 7.5 可视化

```text
fig_C_linec_pareto_3d.svg
fig_C_coupling_vs_noise_delta.svg
fig_C_reservoir_vs_auc_delta.svg
fig_C_signal_spectrum_by_method.svg
fig_C_noop_detection_scatter.svg
fig_C_pareto_fail_reason_heatmap.svg
```

## 7.6 不满足条件时 Codex 先尝试

```text
If CouplingR2 opens but Noise/Reservoir do not:
  stop coupling-only promotion;
  move to reservoir/noise mechanism families only.

If Noise/Reservoir improve but AUC_time worsens:
  add task non-harm projection or reduce event frequency;
  do not lower Noise/Reservoir thresholds.

If all deltas are tiny:
  mark as no-op;
  increase mechanism strength only if holdout and CEp99 allow.

If controls beat functional:
  store as negative evidence;
  do not tune per dataset.
```

---

# 8. Line B：Functional Value Rebuild and Mechanism Validation

## 8.1 目标

在 B320 上重新建立 functional update 的因果证据。v12.12 不再测试“更多 residual variants”，而测试四类机制：

```text
B-M1: AdamW-metric preconditioning / signal-channel SNR update。
B-M2: multi-window reservoir/noise release。
B-M3: function-preserving coordinate maintenance。
B-M4: optimizer-state transport after geometry maintenance。
```

这些都必须与 controls 对比，不能只看自身改善。

## 8.2 Controls

每个机制必须同时比较：

```text
C0 TaskOnlyAdamW
C1 NoOpMatchedOverhead
C2 RandomMatchedNorm
C3 AdamWParallelDirection
C4 SNR-only
C5 GeometryOnlyNoSNR
C6 ShuffledPayload
C7 InvertedPayload
C8 MLPAnalogMaintenance
C9 B320InternalNoOpReparam
```

## 8.3 B-M1：Signal-channel SNR preconditioned AdamW

### 假设

旧 functional additive delta 不能 work，但把 AdamW 自身限制到 population-safe / signal-safe metric 可能 work。

一阶规则：

$$
SNR_k=\mu_k^2-\frac{\sigma_k^2}{b-1}.
$$

定义 gate：

$$
g_k^{new}=g_k\cdot \mathbf{1}[SNR_k>\tau].
$$

或 soft gate：

$$
g_k^{new}=g_k\cdot \sigma\left(\frac{SNR_k-\tau}{T}\right).
$$

### 实验

```text
methods:
  B-M1a hard SNR gate
  B-M1b soft SNR gate
  B-M1c role-wise SNR gate
  B-M1d signal-channel weighted AdamW
  B-M1e SNR + NoiseLeak veto

window:
  one-step, five-step, short-run 3 seeds
```

### 记录指标

```text
SNR_positive_fraction
rolewise_active_fraction
gradient_norm_removed
cos_to_adamw
train_descent
holdout_descent
AUC_step_delta
AUC_time_delta
CouplingR2_delta
NoiseSignalLeak_delta
RealSignalReservoirRatio_delta
control_gap_vs_AdamWParallel
```

### 通过标准

$$
AUC_{time,new}\le AUC_{time,B320},
$$

$$
NoiseSignalLeak_{new}\le NoiseSignalLeak_{B320}-0.01,
$$

$$
control\_gap\ge0.005.
$$

若失败：

```text
If task worsens:
  reduce gate aggressiveness or apply only to high-noise roles.
If no geometry change:
  stop SNR-only; treat it as control, not functional mechanism.
If AdamWParallel matches it:
  no official success.
```

## 8.4 B-M2：Multi-window reservoir/noise release

### 假设

单次 projector solve 不够，但连续多窗口、每次重新估计 $P_{sig},P_{res}$ 可能产生累计 release。

### 实验

```text
B-M2a 5-window projector joint release
B-M2b 10-window low-cost projector release
B-M2c release only when reservoir high
B-M2d release + noise veto
B-M2e release + tail veto
```

事件触发不能用 dataset name，只能用：

```text
RealSignalReservoirRatio high
NoiseSignalLeak high
CEp99 stable
holdout_loss_ratio within bound
cooldown satisfied
```

### 通过标准

5-window 后必须满足：

$$
\sum_{w=1}^{5}\Delta RealSignalReservoirRatio_w\le -0.02,
$$

$$
\sum_{w=1}^{5}\Delta NoiseSignalLeak_w\le -0.02,
$$

$$
\Delta AUC_{time}\le 0,
$$

且 controls 不能解释。

若失败：

```text
If all events rejected:
  loosen trigger only if tail/holdout are safe, not by dataset.
If release < 1e-3 per event:
  stop local projector VJP family.
If release accumulates but AUC worsens:
  add task projection, not loss modification.
```

## 8.5 B-M3：Function-preserving coordinate maintenance

### 假设

当前 additive delta 很难直接改变 signal/reservoir channel。更有希望的是在 logits 几乎不变的条件下改变内部坐标，让后续 AdamW 的训练轨迹更健康。

目标是寻找 $\theta'$：

$$
\|f_{\theta'}(X)-f_{\theta}(X)\|^2\le \epsilon_f,
$$

同时：

$$
Condition(\theta')<Condition(\theta),
$$

$$
RealSignalReservoirRatio(\theta')<RealSignalReservoirRatio(\theta),
$$

$$
NoiseSignalLeak(\theta')\le NoiseSignalLeak(\theta).
$$

### 候选

```text
B-M3a branch scale rebalancing with logit matching
B-M3b direct/quad readout redistribution with logit matching
B-M3c classbranch gain re-centering with function preservation
B-M3d projection P re-centering with output correction
B-M3e low-rank internal coordinate whitening with logit constraint
```

### 记录指标

```text
logit_match_mse
logit_max_abs_drift
KL_before_after
branch_norm_before_after
quad_direct_ratio_before_after
projection_condition_before_after
CouplingR2_delta_after_5_steps
NoiseSignalLeak_delta_after_5_steps
RealSignalReservoirRatio_delta_after_5_steps
AUC_time_delta_after_short_run
optimizer_state_transport_enabled
```

### 通过标准

单次 maintenance 后：

$$
KL(\theta,\theta')\le0.005,
$$

$$
\max|f_{\theta'}-f_{\theta}|\le0.05,
$$

5-step refresh 后：

$$
\Delta RealSignalReservoirRatio\le -0.01,
$$

$$
\Delta NoiseSignalLeak\le -0.01,
$$

$$
\Delta AUC_{time}\le0.
$$

若失败：

```text
If logit drift high:
  solve smaller least-squares correction, not lower gate.
If logit drift low but geometry unchanged:
  coordinate transform is ineffective; stop this family.
If geometry improves but optimizer state breaks:
  move to B-M4 moment transport.
```

## 8.6 B-M4：Optimizer-state transport

### 假设

functional maintenance 改了内部坐标，但 AdamW moments 仍在旧坐标中，导致 P4 short-run 失败。需要把 optimizer state 随 coordinate transform 一起迁移。

### 实验

```text
B-M4a no state transport
B-M4b zero moments after maintenance
B-M4c project moments through same linearized map
B-M4d reset only high-risk roles
B-M4e delayed AdamW restart after maintenance
```

### 记录指标

```text
moment_m_norm_before_after
moment_v_norm_before_after
cos_update_before_after
loss_spike_after_event
AUC_time_after_event
LineC_after_event
bad_step_rate_after_event
```

### 通过标准

$$
bad\_step\_rate\le0.02,
$$

$$
loss\_spike\_count\le control,
$$

$$
LineC\_Pareto\_pass=1,
$$

$$
control\_gap\ge0.005.
$$

---

# 9. Line B 实验阶段

## B0：P3-to-P4 autopsy dataset

### 目标

构造可复用的 P3/P4 对齐表，明确哪些局部指标不预测 P4。

### 输入

```text
v12.10 P4 artifacts
v12.11 P3v2 candidates
v12.11 constrained low-rank solve
B320 anchor traces
```

### 输出 CSV

`v1212_p3_p4_autopsy.csv`

```text
candidate_id
update_type
dataset
seed
p3_old_score
p3_linec_pareto_score
p3_CouplingR2_delta
p3_NoiseSignalLeak_delta
p3_RealSignalReservoirRatio_delta
p3_CEp99_delta
p3_holdout_loss_ratio
p4_acc_delta
p4_AUC_time_delta
p4_ECE_delta
p4_CouplingR2_delta
p4_NoiseSignalLeak_delta
p4_RealSignalReservoirRatio_delta
p4_control_gap
p4_strict_pass
failure_mode
```

### 判断标准

```text
old_score_corr must be reported；
new_score_corr must be reported；
if new_score_corr < 0.30, do not promote any functional family。
```

## B1：Mechanism one-step/five-step probe

### 目标

只验证机制，不进入 full training。

### 设置

```text
base = B320 exact
methods = B-M1, B-M2, B-M3, B-M4 + controls
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2
windows = 1,5,10
```

### Gate

Candidate can enter B2 only if:

$$
\Delta CouplingR^2\ge0.02,
$$

$$
\Delta NoiseSignalLeak\le-0.01,
$$

$$
\Delta RealSignalReservoirRatio\le-0.01,
$$

$$
holdout\_loss\_ratio\le1.01,
$$

$$
CEp99\_delta\le0.05,
$$

$$
control\_gap\ge0.005.
$$

## B2：P4 short-run with strong controls

### 目标

验证 local mechanism 能否转化为 continuous training benefit。

### 设置

```text
base = B320 exact
methods = B1 survivors + controls
seeds = 0,1,2
datasets = MNIST,Fashion-MNIST,KMNIST
short_run_budget = current P4 budget
event_period = fixed, not dataset-specific
```

### Gate

Official functional short-run pass if:

$$
Acc_{func}\ge Acc_{B320}-0.003,
$$

$$
AUCtime_{func}\le AUCtime_{B320},
$$

$$
ECE_{func}\le ECE_{B320}+0.01,
$$

$$
CEp99_{func}\le CEp99_{B320}+0.05,
$$

$$
CouplingR^2_{func}\ge CouplingR^2_{B320}+0.02,
$$

$$
NoiseSignalLeak_{func}\le NoiseSignalLeak_{B320}-0.01,
$$

$$
RealSignalReservoirRatio_{func}\le RealSignalReservoirRatio_{B320}-0.01,
$$

$$
amortized\_overhead\le1.05,
$$

$$
control\_gap\ge0.005.
$$

## B3：3-seed confirm

Only if B2 passes.

```text
methods:
  B320 AdamW
  B320 NoOp overhead
  B320 AdamWParallel
  B320 SNR-only
  B320 best functional
  MLP AdamW
  MLP analog maintenance

seeds = 0,1,2
```

If B3 fails, do not expand to 10 seeds.

---

# 10. Line D：Classic No-BSpline Family Closure

## 10.1 总原则

Line D 不能抢 Line B 资源，但必须继续产出明确 family status。B-spline 本版继续冻结。

Active families:

```text
Rational
Chebyshev
Wavelet
RBF/FastKAN
Fourier
```

每个 family 的输出状态只能是：

```text
FamilyPass
FamilyNearPass
KernelBlocked
ExpressionBlocked
TaskBlocked
GeometryBlocked
RejectedForThisVersion
```

## 10.2 Rational：TaskBlocked by A5 trajectory

### 当前事实

Rational 已测试：

```text
B7kc/B7kd/B7ke bounded rational hidden residual + gain bracket
B7kf/B7kg/B7kh tanh residual + stop-gradient cap bracket
B7lq/B7lr pairNorm + pair-signal isolation
```

共同结果：

```text
L3 efficiency pass = 1
A4 expression pass = 1
A5 task pass = 0
Line C not obviously bad 或 reservoir high，但不是 gate opener
```

### v12.12 只允许一个新方向

不要继续降低 cap/gain。只测试一个 structural trajectory family：

```text
RAT-T1 Rational trajectory optimizer / moment schedule
RAT-T2 rational group warmup then release
RAT-T3 B320-like classbranch gain transport into Rational
```

### 记录指标

```text
rational_den_p01
r_prime_p95
r_double_prime_p95
group_function_diversity
AUC_step_ratio
AUC_time_ratio
ECE_delta
CEp99_delta
CouplingR2_delta
NoiseSignalLeak_delta
RealSignalReservoirRatio_delta
```

### 通过标准

$$
A5=1,
$$

or at least:

$$
mean\_delta\ge -0.003,
$$

$$
worst\_delta\ge -0.02,
$$

$$
near\_pass\_rate\ge0.80,
$$

$$
AUC\_step\_ok=1.
$$

If still fail:

```text
Rational = TaskBlocked_RejectedForThisVersion
```

## 10.3 Chebyshev：TaskBlocked after A4 open

### 当前事实

B3ao 是当前最接近的 Chebyshev candidate，但仍：

```text
A4 = 1
A5 = 0
mean_delta ≈ -0.0556
worst_delta ≈ -0.09375
near_pass_rate = 0
RealSignalReservoirRatio_delta_vs_mlp ≈ 0.573
```

### 新方向

不要继续 raw-linear strength scaling。测试：

```text
CHEB-G1 degree-energy damping
CHEB-G2 low-degree dominant with signal readout
CHEB-G3 degree whitening + reservoir veto
CHEB-G4 Cheby-to-B320 coordinate hybrid diagnostic
```

### 通过标准

```text
A4 must remain pass；
mean_delta improves by >= 0.03 over B3ao；
RealSignalReservoirRatio_delta reduces by >= 0.20；
near_pass_rate > 0；
AUC_step_ok improves。
```

If no improvement:

```text
Chebyshev = TaskBlocked_RejectedForThisVersion
```

## 10.4 Wavelet：TaskBlocked after A4 open

### 当前事实

Wavelet progressed from ExpressionBlocked to TaskBlocked:

```text
B5h baseline A4 fail；
B5i/B5j/B5k A4 pass；
B5m/B5n/B5p/B5q task improves but still A5 fail；
B5n/B5p/B5q plateau around mean_delta ≈ -0.055 to -0.057, worst ≈ -0.076。
```

### 新方向

不要继续 raw-linear scale. Test one structural family:

```text
WAV-G1 multi-scale low/high support balance
WAV-G2 local-to-global sidecar with Line C reservoir veto
WAV-G3 scale-energy damping functional diagnostic
```

### 通过标准

```text
A4 pass maintained；
mean_delta improves by >= 0.03 over B5n；
worst_delta >= -0.03；
near_pass_rate >= 0.50 exploratory；
Line C nontearing。
```

If no improvement:

```text
Wavelet = TaskBlocked_RejectedForThisVersion
```

## 10.5 RBF/FastKAN：ExpressionBlocked after fused L3

### 当前事实

RBF now has real fused L3 efficiency but A4 remains blocked:

```text
B2r/B2s fused L3 pass but A4 fail；
input-cross B2t/B2u/B2v improves E1/E6/E8 but still fail；
rank256 B2w/B2x/B2y does not solve E6/E8；
orthoproj B2z/B2aa/B2ab also fails。
```

### New direction

Do not increase fixed sidecar rank. Test one of:

```text
RBF-G1 learned quantile centers with fixed width and fused path
RBF-G2 orthogonal local basis + trainable center scale
RBF-G3 RBF + Cheby low-degree hybrid diagnostic
```

### Gate

```text
L3 efficiency must remain pass；
A4 must pass；
if A4 does not pass, do not run A5；
memory_ratio <= 1.05。
```

If A4 still fail:

```text
RBF = ExpressionBlocked_RejectedForThisVersion
```

## 10.6 Fourier：ExpressionBlocked

### 当前事实

Fourier has excellent efficiency but A4 fail, and higher K worsens E6/E8. Therefore do not increase order.

### New direction

```text
FOU-G1 lowfreq + input-cross sidecar with orthogonal projection
FOU-G2 localized Fourier window diagnostic
FOU-G3 Fourier-Cheby hybrid low-degree diagnostic
```

### Gate

```text
A4 pass required before A5；
NoiseSignalLeak must not increase；
high_frequency_energy_ratio <= 0.30 unless expression target requires it。
```

If A4 still fail:

```text
Fourier = ExpressionBlocked_RejectedForThisVersion
```

## 10.7 Line D artifacts

```text
v1212_family_manifest.csv
v1212_family_efficiency.csv
v1212_family_gradcheck.csv
v1212_family_expression.csv
v1212_family_task_triage.csv
v1212_family_linec.csv
v1212_family_failure_table.csv
v1212_family_status.json
```

## 10.8 Line D 可视化

```text
fig_D_family_status_matrix.svg
fig_D_family_efficiency_expression_pareto.svg
fig_D_family_task_blocker_waterfall.svg
fig_D_family_linec_radar.svg
fig_D_family_expression_delta_heatmap.svg
fig_D_family_task_vs_reservoir.svg
```

---

# 11. 并行执行安排

为了加快进度，v12.12 分成三个并行 batch。

## Batch 1：立即可跑，主要复用 artifacts

```text
A0 B320 exact hardening summary rebuild
B0 P3-to-P4 autopsy table
C0 Line C value rule implementation
D0 family status freeze/rewrite
```

预期耗时短，因为主要是 artifact analysis / wrapper。

## Batch 2：机制探针并行

```text
B-M1 SNR / signal-channel AdamW preconditioner
B-M2 multi-window reservoir/noise release
B-M3 function-preserving coordinate maintenance
B-M4 optimizer-state transport
```

每个机制都跑 MNIST/Fashion/KMNIST seeds 0,1,2，但先只 one-step/five-step，不进 short-run。

## Batch 3：classic family limited repair

```text
Rational one structural trajectory attempt
Chebyshev one task-geometry family
Wavelet one task-geometry family
RBF one expression repair family
Fourier one expression repair family
```

每个 family 最多一个 structural direction，不再无限小网格。

---

# 12. v12.12 最终 route 规则

v12.12 最终只能输出以下 route 之一。

```text
R1-B320AnchorLockedFunctionalParetoPass
  B320 locked, functional B1/B2 Pareto pass, ready for B3 3-seed confirm.

R2-B320AnchorLockedFunctionalValueStillInvalid
  B320 locked, but no functional value score predicts P4 or no candidate P3v2 pass.

R3-B320AnchorLockedLocalDirectionsInsufficient
  Coupling opens, but NoiseSignalLeak / Reservoir cannot be moved by local directions.

R4-B320AnchorRegression
  B320 fails hardening; must fix anchor before functional.

R5-ClassicFamilyCandidateEmerges
  Some non-B-spline classic family passes L3/A4/A5/LineC; compare against B320.

R6-ClassicFamiliesRejectedForThisVersion
  Non-B-spline families have clear blockers; continue B320 functional only.
```

---

# 13. v12.12 成功标准

## 13.1 Minimum success

```text
B320 anchor locked；
旧 P3 score 被正式废弃；
新 Line C Pareto value 实现；
functional 机制至少有一个通过 one-step/five-step Pareto；
classic family statuses 全部明确。
```

## 13.2 Strong success

```text
B320 best functional 通过 P4 short-run；
control_gap >= 0.005；
AUC_time 不坏；
NoiseSignalLeak 与 RealSignalReservoirRatio 同时下降；
amortized overhead <= 1.05。
```

## 13.3 Paper-relevant success

```text
B320 + functional 在 same base 上超过 B320 + AdamW；
beats AdamWParallel / RandomMatchedNorm / SNR-only / NoOp / MLP analog；
Line C 解释 functional 改善来自 signal/reservoir/noise geometry，而不是 task-only drift；
速度/显存仍保持 MLP-like 或 better。
```

---

# 14. 最终判断

v12.11 是关键进展，不是原地失败：

```text
1. B320 anchor 已锁定。
2. functional 旧 score 被证伪。
3. CouplingR2 不再是核心难点。
4. NoiseSignalLeak 和 RealSignalReservoirRatio 才是 functional 机制硬瓶颈。
5. Rational/Chebyshev/Wavelet/RBF/Fourier 支线都有明确 blocker，不能再盲扫。
```

v12.12 的核心不是继续扩大候选，而是减少自由度：

$$
\boxed{
\text{固定 B320，废弃旧 P3 score，重建 Line C Pareto value，}
\text{只验证能真正移动 Noise/Reservoir 且不伤 task 的机制。}
}
$$

如果 v12.12 仍然证明局部 functional 方向无法移动真实 NoiseSignalLeak / Reservoir，那么下一步就必须承认：当前形式的 functional update 不适合作为训练期 additive maintenance；它应转为更慢、更结构性的 coordinate reparameterization，或作为 post-training geometry tool，而不是继续伪装成 online optimizer improvement。

