# DG-KAN v12.13：B320 Locked Base 上的 Functional 机制重建与 No-BSpline 经典基函数支线计划

> 版本：v12.13 execution plan  
> 基于：v12.12 `B320Locked FunctionalValueRebuild ClassicNoBSpline` 执行复盘  
> 目标：不再把精力放在寻找新 base 或继续小修 F14/F15/BM3 强度，而是在已经锁定的 B320 base 上重建 functional update 的 value source；同时保留 No-BSpline 经典基函数支线，但不让它抢占主线资源。  
> 公式格式：Typora 友好，只使用 `$...$` 与 `$$...$$`。  
> 硬约束：CE-only；no teacher；no distillation；no loss modification；no sampler / class weight；no dataset-name branch；no fake / proxy / CPU offload；base path 不使用 PyTorch `loss.backward()` 作为 official 更新路径；functional 不能用 validation/test 或 future outcome 作为 commit-time 特征。

---

## 0. 一句话结论

v12.12 不是没有进展。它完成了一个很重要的阶段转换：

$$
\boxed{
\text{B320 base 已经锁定；functional update 的旧 value score 被证伪；}
\text{现在真正的问题是 functional 机制本身。}
}
$$

这和前面几轮完全不同。过去的主要问题是：KAN base 是否足够快、是否表达力不打折、是否训练轨迹健康。现在 B320 已经满足这些条件，至少在当前 MNIST / Fashion-MNIST / KMNIST 小样本严格协议上，B320 已经是一个可用 anchor。

当前最核心的事实是：

```text
B320_anchor_locked = 1
step_ratio_q90 = 0.39931987348441034
memory_ratio_q90 = 0.1295238095238095
mean_delta = 0.027669270833333332
worst_delta = -0.001953125
near_pass_rate = 1.0
AUC_step_ratio = 0.9336784156141542
AUC_time_ratio = 0.7424718100091173
ECE_delta = 0.01767905056476593
LineC_nontearing_pass = 1
```

这说明 base 主线已经不应该继续“小修 B320”。B320 的角色应该从 candidate 变成 fixed anchor。

但是 functional update 仍然没有成功。v12.12 的核心负结果是：

```text
p4_strict_pass_rows = 0
linec_pareto_pass_rows = 0
do_not_promote_functional = 1
pareto_pass_rows = 0
noise_release_rows = 0
reservoir_release_rows = 3
no_op_rows = 102
```

因此 v12.13 的核心路线是：

$$
\boxed{
\text{锁定 B320，不再找 base；}
\text{把 functional update 从 coupling-only move 重建为 signal-reservoir/noise-safe constrained update。}
}
$$

---

## 1. 这次到底有没有进展

### 1.1 有，而且是关键进展

v12.12 的第一类进展是 **B320 anchor 真实闭合**。B320 的 step ratio 已经远低于 MLP-like gate，memory ratio 非常低，task mean delta 为正，near pass 为 1.0，AUC-step 和 AUC-time 都优于 MLP，Line C nontearing 也通过。这意味着我们终于有了一个可以承载 functional official re-entry 的 strict FC-PureKAN base。

这件事的意义很大：从 v12.0 到 v12.6，我们一直在 base、efficiency、expression、task trajectory 之间来回卡住；现在这些不再是主 blocker。

### 1.2 但这不是 functional success

v12.12 的 functional value autopsy 说明，旧 P3 score 和 P4 gain 基本没有可用预测关系：

```text
old_score_spearman_to_p4_gain = 0.06405518470831038
new_linec_score_spearman_to_p4_gain = 0.18625523432899613
old_score_pearson_to_p4_gain = -0.007065898088334216
new_linec_score_pearson_to_p4_gain = 0.012352855744075323
```

Line C score 的 Spearman 相关略高，但 Pearson 仍接近 0，而且没有任何 hard Pareto pass。这说明：

$$
\boxed{
\text{旧 value score 不能作为 promotion score；}
\text{新的 Line C rule 是更好的诊断语言，但还没有产生可 promotion 的 functional update。}
}
$$

### 1.3 BM1 给出 partial positive，但不是 survivor

B-M1 的 SNR / signal-channel probe 有真实价值。它证明在某个具体 slice 上，SNR gate 可以同时改善 coupling、noise leak 和 reservoir ratio。例如 Fashion-MNIST seed 0 / window 5 上，BM1b-SoftSNRGate 做到了：

```text
CouplingR2_delta = 0.16610733182736603
NoiseSignalLeak_delta = -0.011934816837310791
RealSignalReservoirRatio_delta = -0.014312267303466797
control_gap_vs_best = 0.0889776448538202
```

这说明 SNR 方向不是完全没信号。但它只在 1/9 row 上成立：

```text
bm1_rows = 54
bm1_pareto_pass_rows = 3
bm1_full_3x3_pareto_candidate_count = 0
bm1_partial_pareto_candidates = BM1a-HardSNRGate@window5:1/9,BM1b-SoftSNRGate@window5:1/9,BM1d-SignalWeightedAdamW@window5:1/9
```

所以它是：

$$
\boxed{
\text{机制存在的局部证据，不是可部署 functional update。}
}
$$

下一步不能把 BM1 直接推广成 P4 short-run，而要问：为什么它只在 Fashion-MNIST seed 0 / window 5 成立？它的有效条件是数据局部结构、window size、SNR threshold，还是 batch sketch 噪声？

### 1.4 BM3 证明 coupling 容易被“刷高”，但这不等于几何好

B-M3a branch rebalance 的结果非常有启发。它可以把 CouplingR2_delta 做得很大，例如 KMNIST seed 2 上：

```text
CouplingR2_delta = 0.7698809892042978
CEp99_delta = -0.057579994201660156
```

但同一 row 里：

```text
logit_max_abs_drift = 0.13549089431762695
NoiseSignalLeak_delta = 0.0024562478065490723
RealSignalReservoirRatio_delta = -0.005616992712020874
fail_reason = logit_max_abs_drift>0.05;NoiseSignalLeak_delta>-0.01;RealSignalReservoirRatio_delta>-0.01
```

这说明 branch rebalance 更多是 coordinate move：它可以改变 train-probe displacement 的线性可预测性，但没有真正把噪声从 signal channel 移走，也没有稳定释放 reservoir 中的真实信号。

因此 BM3 的定位必须下调：

```text
不是 functional value source；
可以作为 coordinate transport / optimizer-state transport 的辅助工具；
不能单独 promotion。
```

---

## 2. 为什么仍然感觉慢

你感觉慢是合理的，因为 functional update 这条线还没有出现 official survivor。但这不是“没有进度”，而是项目进入了更难的层级。

过去我们在解决：

```text
KAN base 是否能跑得像 MLP；
KAN base 是否表达力不打折；
KAN base 是否训练轨迹不比 MLP 差。
```

现在变成：

```text
在一个已经合格的 B320 PureKAN base 上，
functional update 是否能给出独立、可预测、control-resistant 的几何收益？
```

这是更难的问题。因为普通 task descent、AdamWParallel、SNR-only、NoOp / RandomMatchedNorm 都可能解释一部分收益。我们不能只看到 CouplingR2 提高就宣布成功。

现在的 functional gate 实际要求同时满足：

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
\Delta AUC_{time} \le 0,
$$

$$
control\_gap \ge 0.005,
$$

并且 task、ECE、CEp99、logit drift、overhead 不能坏。v12.12 数据说明：当前多数方向只能满足其中一部分。

---

## 3. 当前真正卡在哪里

### 3.1 不是 base 卡住

B320 的 base 条件已经非常强：

```text
step_ratio_q90 = 0.3993
memory_ratio_q90 = 0.1295
mean_delta = +0.0277
near_pass = 1.0
AUC_time = 0.7425
LineC_nontearing = 1
```

因此 v12.13 不应再把主资源投到 B314/B320 小修。B320 只做 hardening monitor，不做主优化对象。

### 3.2 不是 CouplingR2 卡住

Line C value rule 里：

```text
coupling_open_rows = 92 / 182
```

BM3 甚至可以让 CouplingR2_delta 接近 0.77。说明 CouplingR2 已经不是最稀缺的东西。真正稀缺的是：

```text
noise_release_rows = 0
reservoir_release_rows = 3
```

也就是说：

$$
\boxed{
\text{functional update 现在会“动”，也会提高 coupling，}
\text{但不会稳定做到 noise release 与 reservoir release。}
}
$$

### 3.3 核心 blocker：functional value 没有对准 population-safe direction

根据 signal-channel / reservoir 的语言，好的 functional update 不只是让 train motion 更像 probe motion，而是要做到：

```text
真实任务信号进入 signal channel；
噪声不要进入 signal channel；
已经在 reservoir 里的真实信号被释放；
hard tail 不变坏；
task trajectory 不变慢。
```

当前 BM1 和 BM3 各自暴露了一个问题：

```text
BM1:
  有局部 signal/noise/reservoir 同向改善，但覆盖太低。

BM3:
  coupling 容易打开，但 noise/reservoir 没有同步改善。
```

所以 v12.13 要解决的不是“再调一个 lambda”，而是构造一个真正的 constrained functional solve：更新方向必须直接以 noise leak 和 reservoir release 为约束，而不是只优化 coupling 或 branch scale。

---

## 4. 是否在正确道路上

我认为方向是正确的，而且比前几轮更清楚。正确的地方包括：

```text
1. B320 已经锁定，不再把未合格 base 强行交给 functional。
2. 没有把 P3 positive 写成 P4 success。
3. 没有把 BM1 的 1/9 partial pass 写成 survivor。
4. 没有把 BM3 的高 CouplingR2 写成 geometry success。
5. 没有按 MNIST/Fashion/KMNIST 做 dataset-specific controller。
6. B-spline 已冻结，classic family 不再无底洞式消耗主线资源。
```

需要改变的是：functional update 不能再沿旧 F14/F15、branch damping、residual weight 小网格继续做。那些已经证明要么 P3-to-P4 不可转化，要么被 controls 吃掉。

v12.13 应当进入：

$$
\boxed{
\text{Functional value source reconstruction。}
}
$$

---

## 5. 离目标还差多远

### 5.1 离“合格 PureKAN base”已经很近，当前可认为主线已锁定

在当前协议下，B320 已经是 strong anchor。还需要更宽协议确认，但不应继续把 base 搜索当主目标。

### 5.2 离“functional update 成功”仍然远

当前仍是：

```text
official_functional_success = 0
p4_strict_pass_rows = 0
linec_pareto_pass_rows = 0
```

并且 BM1 / BM3 只说明局部机制存在或 coupling 可动，不能说明 functional update 有独立价值。

### 5.3 离“next-gen MLP claim”还差一个核心闭环

最终 claim 必须是：

```text
B320 + functional update
在同一 base、同一 CE-only、同一数据、同一系统预算下，
比 B320 + ordinary AdamW 更快 / 更稳 / 几何更好，
并击败 AdamWParallel、RandomMatchedNorm、SNR-only、NoOp、MLP analog controls。
```

现在缺的是这一段。

---

# 6. v12.13 总体目标

v12.13 的目标不是继续找 base，也不是继续试所有旧 functional 小变体。目标是：

$$
\boxed{
\text{在 B320 locked anchor 上，构造并验证一个真正以 Line C Pareto 为目标的 functional mechanism。}
}
$$

具体分成四条线：

```text
Line A：B320 Anchor Monitor
  保持 B320 不变，只做复现、协议扩展和 no-regression 检查。

Line C：Manifold-Channel Geometry Diagnostics Calibration
  先确认 NoiseSignalLeak / RealSignalReservoirRatio 的测量噪声、阈值和 repeatability。

Line B：Functional Mechanism Rebuild
  不再继续 F14/F15 小网格；只测试 BM1/BM2/BM3/BM4/BM5 机制族。

Line D：Classic No-BSpline Family Closure
  Rational / Chebyshev / Wavelet / RBF / Fourier 继续；B-spline 冻结。
```

四条线的优先级是：

```text
B320 locked base 是地基；
Line C 是价值定义；
Line B 是主科学问题；
Line D 是并行 portfolio，不得抢主线资源。
```

---

# 7. Line A：B320 Anchor Monitor

## 7.1 目标

确认 B320 是稳定 anchor，而不是继续优化它。Line A 不允许新增 B320 小修候选，除非出现复现失败。

## 7.2 假设

$$
H_A:
\text{B320 在当前协议扩展下仍保持 task/efficiency/Line C nontearing pass。}
$$

## 7.3 实验设计

固定 candidate：

```text
B320b-SimpleFastTaskGeometry-h160-learnableP-trainProbeP-signalBroad035-signalBlock015-quadReadInit125-directRamp105-direct050-fusedProjGradAdamW-absdiag050-manualAdamW-classbranch-fixedgain-gainramp075-identitytailquad030-hingeamp025-temp075
```

运行：

```text
datasets = MNIST, Fashion-MNIST, KMNIST
seeds = 0..9 for exact reconfirm
optional seeds = 10..19 for stress monitor
train_size = 1024 baseline; optional 2048 stress
val_size = 512
test_size = 512
epochs = 3 baseline; optional 5 stress
batch_size = 128
```

## 7.4 必须记录

```text
candidate_id
run_id
dataset
seed
train_size
val_size
test_size
epoch
step_ratio_q90
memory_ratio_q90
forward_ratio_q90
backward_ratio_q90
update_ratio_q90
val_acc_delta_vs_mlp
AUC_step_ratio
AUC_time_ratio
ECE_delta
NLL_delta
CEp99_delta
margin_p10_delta
CouplingR2
NoiseSignalLeak
RealSignalReservoirRatio
LineC_nontearing_pass
strict_pass
fail_reason
```

## 7.5 判断标准

B320 anchor 保持 locked：

$$
step\_ratio_{q90} \le 0.60,
$$

$$
memory\_ratio_{q90} \le 0.20,
$$

$$
mean\_delta \ge 0,
$$

$$
worst\_delta \ge -0.003,
$$

$$
near\_pass\_rate = 1.0,
$$

$$
AUC_{step} \le 1.00,
$$

$$
AUC_{time} \le 1.00,
$$

$$
LineC_{nontearing}=1.
$$

## 7.6 不满足时 Codex 先尝试什么

如果 B320 复现失败：

```text
1. 先检查 timing warmup / compile / artifact reuse，不改模型。
2. 检查 MLP baseline 是否漂移。
3. 检查 fixedgain/global gain init 是否和 v12.12 完全一致。
4. 检查 F3 fused kernel path 是否退回非 official path。
5. 只在确认实现回归后修 implementation，不引入新 candidate。
```

如果只在某个 dataset/seed 失败：

```text
只做 failure slice autopsy；
不得新增 dataset-specific rule；
不得按 Fashion/KMNIST/MNIST 调阈值。
```

---

# 8. Line C：Manifold-Channel Geometry Diagnostics Calibration

## 8.1 为什么必须先校准 Line C

v12.12 中所有 182 行没有 hard Pareto pass，尤其是：

```text
NoiseSignalLeak_delta>-0.01: 182 / 182
RealSignalReservoirRatio_delta>-0.01: 179 / 182
```

这可能有两种解释：

```text
解释 A：functional 机制确实没有 noise/reservoir 作用。
解释 B：当前 batch/sketch/window 下，NoiseSignalLeak 和 Reservoir 的估计方差太大，阈值不可达或不可重复。
```

不能直接假设 A。v12.13 必须先测 Line C 指标的 null distribution、repeatability、effect-size floor。

## 8.2 假设

$$
H_C0:
\text{Line C 的 NoiseSignalLeak 与 RealSignalReservoirRatio 在 B320 上可重复测量，且阈值 }0.01\text{ 大于测量噪声。}
$$

## 8.3 实验设计

在同一 B320 checkpoint 上跑：

```text
NoOp repeated
TaskOnlyAdamW repeated
AdamWParallelDirection repeated
RandomMatchedNorm repeated
BM1b repeated
BM3a repeated
```

对每个 method：

```text
datasets = MNIST, Fashion-MNIST, KMNIST
seeds = 0,1,2
functional_batch_size = 16,32,64
probe_splits = 5 repeated splits
window = 3,5,10
sketch_dim = 32,64,128
ridge_lambda = 1e-3,1e-2,1e-1
```

## 8.4 必须记录

```text
run_id
candidate_id
method
dataset
seed
functional_batch_size
probe_split_id
window
sketch_dim
ridge_lambda
CouplingR2_before
CouplingR2_after
CouplingR2_delta
NoiseSignalLeak_before
NoiseSignalLeak_after
NoiseSignalLeak_delta
RealSignalReservoirRatio_before
RealSignalReservoirRatio_after
RealSignalReservoirRatio_delta
CEp99_delta
ECE_delta
AUC_proxy_delta
logit_max_abs_drift
holdout_loss_ratio
bootstrap_ci_low
bootstrap_ci_high
metric_std_across_splits
noop_false_positive
random_false_positive
control_gap_vs_best
```

## 8.5 判断标准

Line C calibration pass：

$$
|\mu_{NoOp}(\Delta NoiseSignalLeak)| \le 0.003,
$$

$$
\sigma_{NoOp}(\Delta NoiseSignalLeak) \le 0.005,
$$

$$
|\mu_{NoOp}(\Delta RealSignalReservoirRatio)| \le 0.003,
$$

$$
\sigma_{NoOp}(\Delta RealSignalReservoirRatio) \le 0.005.
$$

Functional effect must be measured with CI:

$$
CI_{low}(\Delta NoiseSignalLeak) \le -0.01,
$$

$$
CI_{low}(\Delta RealSignalReservoirRatio) \le -0.01.
$$

如果 NoOp 或 RandomMatchedNorm 在当前 metric 下频繁过 gate，则 Line C metric calibration 失败，不能进入 P4。

## 8.6 不满足时 Codex 先尝试什么

如果 metric 方差太大：

```text
1. functional_batch_size 从 16 提到 32/64。
2. sketch_dim 从 32 提到 64/128。
3. 固定 B/Q split 做 paired measurement。
4. 增加 probe_splits 并使用 bootstrap CI gate。
5. 不降低阈值；先报告 noise floor。
```

如果 CouplingR2 重复性好，但 Noise/Reservoir 不好：

```text
CouplingR2 只能作为辅助指标；
functional promotion 必须等待 Noise/Reservoir metric 稳定。
```

---

# 9. Line B：Functional Mechanism Rebuild

## 9.1 总目标

在 B320 fixed anchor 上，构造一个满足 Line C Pareto 的 functional update：

$$
\Delta\theta_{new}=\Delta\theta_{task}+\lambda\Delta\theta_{func}.
$$

其中：

```text
Delta_task:
  B320 原 manual AdamW / fusedProjGradAdamW task update。

Delta_func:
  低频、task-safe、control-resistant 的 signal-reservoir/noise-safe correction。
```

## 9.2 不再继续的方向

v12.13 不再继续：

```text
F14/F15 lambda 小网格；
branch damping 小权重 sweep；
只优化 CouplingR2 的 branch rebalance；
不带 noise/reservoir 约束的 residual/orthogonal geometry；
任何 dataset-specific controller；
把 single-row partial positive 写成 survivor。
```

## 9.3 Functional 候选机制

### BM1-ext：Multi-window SNR / Signal-channel gate

BM1 只在 Fashion-MNIST seed 0 / window 5 上出现局部成功。v12.13 不直接 promotion，而是扩展成 multi-window、role-wise、CI-gated 版本。

候选：

```text
BM1e-SoftSNRGate-window3
BM1f-SoftSNRGate-window5
BM1g-SoftSNRGate-window10
BM1h-RoleWiseSNR-direct
BM1i-RoleWiseSNR-quad
BM1j-RoleWiseSNR-branch
BM1k-WindowEnsembleSNR
BM1l-SNRGateWithNoiseVeto
```

核心公式：

$$
SNR_k=\mu_k^2-\frac{\sigma_k^2}{b-1}.
$$

接受的参数方向：

$$
g'_k = g_k \cdot \mathbb{1}[SNR_k > \tau].
$$

soft 版本：

$$
g'_k = g_k \cdot \sigma\left(\alpha \frac{SNR_k-\tau}{|\tau|+\epsilon}\right).
$$

判断标准：

```text
不能只在 1/9 row 通过；
必须在 3x3 全部 row 通过，或 8/9 通过且失败 row 不能集中在单一 dataset；
NoiseSignalLeak_delta 和 RealSignalReservoirRatio_delta 必须 CI 有效；
control_gap_vs_best >= 0.005。
```

失败后 Codex 先尝试：

```text
1. 如果只在 Fashion seed0 有效：检查 SNR_positive_fraction 与 gradient_norm_removed 分布，做 window ensemble，不做 Fashion-specific threshold。
2. 如果 holdout_loss_ratio 好但 noise 不释放：增加 NoiseLeakVeto，而不是加大强度。
3. 如果 cos_to_adamw > 0.95 且 control_gap 小：加入 AdamW-orthogonal residual。
4. 如果 coverage 太低：改为 role-wise SNR，不改 dataset threshold。
```

### BM2：Constrained Signal-Reservoir Projector

BM2 是 v12.13 的主机制候选。它不再先构造一个 functional update 再看 Line C，而是直接把 Line C 作为约束。

在低秩子空间 $U$ 中求：

$$
d = U a.
$$

目标：

$$
\max_a
\quad
\alpha \Delta CouplingR^2(a)
-
\beta \Delta NoiseSignalLeak(a)
-
\gamma \Delta RealSignalReservoirRatio(a)
-
\eta \Delta CEp99(a).
$$

约束：

$$
holdout\_loss\_ratio(a) \le 1.002,
$$

$$
logit\_max\_abs\_drift(a) \le 0.05,
$$

$$
CEp99\_delta(a) \le 0.05,
$$

$$
\|d\| \le \rho\|d_{AdamW}\|.
$$

子空间来源：

```text
U1: AdamW-orthogonal gradient residual basis
U2: per-example gradient SNR-positive basis
U3: signal projector basis from Line C sketch
U4: reservoir-release VJP basis
U5: noise-leak veto basis
U6: role-wise direct/quad/branch basis concatenation
```

候选：

```text
BM2a-LowRankLineCQP-U1
BM2b-LowRankLineCQP-U2
BM2c-LowRankLineCQP-U3
BM2d-LowRankLineCQP-U4
BM2e-LowRankLineCQP-U1U3
BM2f-LowRankLineCQP-U2U4
BM2g-LineCQP-withNoiseVeto
BM2h-LineCQP-withTailVeto
```

必须记录：

```text
subspace_id
rank
alpha beta gamma eta
constraint_active_count
solve_status
solve_time_ms
functional_norm_ratio
adamw_cosine
task_orthogonal_fraction
CouplingR2_delta
NoiseSignalLeak_delta
RealSignalReservoirRatio_delta
CEp99_delta
ECE_delta
margin_p10_delta
holdout_loss_ratio
logit_max_abs_drift
control_gap_vs_best
amortized_overhead_ratio
```

通过条件：

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
holdout\_loss\_ratio \le 1.002,
$$

$$
logit\_max\_abs\_drift \le 0.05,
$$

$$
control\_gap \ge 0.005.
$$

失败后 Codex 先尝试：

```text
1. 如果 CouplingR2 过但 noise 不过：提高 beta，不提高 alpha。
2. 如果 reservoir 不过：增加 U4 reservoir-release basis 或提高 gamma。
3. 如果 logit drift 过大：加入 logit least-squares preservation constraint。
4. 如果 solve 太慢：rank 从 32 降到 16/8，或预计算 basis。
5. 如果 control_gap 小：移除与 AdamW 平行部分，做 task-orthogonal residual。
```

### BM3-ext：Branch Rebalance with Logit LS Constraint

BM3 当前问题是 coupling-only。v12.13 只允许它作为 coordinate maintenance，不允许单独 promotion。

新增约束：

$$
\min_{\Delta b,\Delta s}
\|f_{\theta+\Delta}(B)-f_{\theta}(B)\|_2^2
+
\lambda\|\Delta\|_2^2,
$$

同时要求：

$$
logit\_max\_abs\_drift \le 0.05.
$$

候选：

```text
BM3b-BranchRebalanceLogitLS-s0.005
BM3c-BranchRebalanceLogitLS-s0.010
BM3d-BranchRebalanceLogitLS-withNoiseVeto
BM3e-BranchRebalanceLogitLS-plusBM1SNR
BM3f-BranchRebalanceTransportOnly
```

BM3 进入 P4 的条件比 BM2 更严格：它必须证明不是 coupling-only：

$$
\Delta NoiseSignalLeak \le -0.01
$$

且：

$$
\Delta RealSignalReservoirRatio \le -0.01.
$$

否则只作为 BM4 的 state transport 前置。

### BM4：Optimizer-State / Moment Transport

BM3 暴露一个可能问题：coordinate 变化后，AdamW moments / fusedProjGradAdamW state 没有同步，导致 P3 改善无法转成 P4 continuous training。

BM4 测试：

```text
BM4a-CoordinateChange-NoTransport
BM4b-CoordinateChange-MomentReset
BM4c-CoordinateChange-MomentProjected
BM4d-CoordinateChange-MomentWhitened
BM4e-BM2Update-withMomentProjected
BM4f-BM1Update-withMomentProjected
```

必须记录：

```text
moment_transport_type
m_norm_before_after
v_norm_before_after
update_cos_before_after
AUC_proxy_delta
CouplingR2_delta
NoiseSignalLeak_delta
RealSignalReservoirRatio_delta
bad_step_rate_after_event
```

通过条件：BM4 自身不要求创造 value，但必须降低 P3-to-P4 mismatch：

$$
P4Gain_{transport} - P4Gain_{no\_transport} \ge 0.005.
$$

失败后 Codex 先尝试：

```text
1. 如果 moment reset 更好：functional event 后短 cooldown。
2. 如果 projected 更好：只 transport active roles。
3. 如果所有 transport 都不改善：说明 P3-P4 mismatch 不是 optimizer-state，而是 functional direction 本身。
```

### BM5：Reservoir-targeted VJP Correction

BM5 直接针对最难的 `RealSignalReservoirRatio`。

定义真实 residual：

$$
r_{real}=y-f_\theta(x).
$$

reservoir 部分：

$$
r_{res}=P_{res}r_{real}.
$$

构造使 $r_{res}$ 下降的 VJP update：

$$
d_{res}=-J^T P_{res} r_{real}.
$$

同时对噪声 signal leak 做 veto：

$$
\langle Jd_{res}, P_{sig}r_{noise}\rangle \le 0.
$$

候选：

```text
BM5a-ReservoirVJP
BM5b-ReservoirVJP-AdamWOrthogonal
BM5c-ReservoirVJP-NoiseVeto
BM5d-ReservoirVJP-TailSafe
BM5e-ReservoirVJP-LowRankQP
```

如果 BM5 可以稳定降低 RealSignalReservoirRatio，但 task/AUC 变坏，则说明 reservoir release 需要更强 task-safe projection；进入 BM2 constrained QP，不做 standalone。

---

# 10. P3 Functional Mechanism Gate

## 10.1 运行矩阵

所有 BM candidates 在 cloned checkpoint 上运行，不污染真实训练：

```text
datasets = MNIST, Fashion-MNIST, KMNIST
seeds = 0,1,2
functional_batch_size = 32 baseline, 64 confirm
window = 5 baseline; 3/10 diagnostic
controls = TaskOnlyAdamW, NoOpMatchedOverhead, RandomMatchedNorm, AdamWParallelDirection, SNR-only, GeometryOnlyNoSNR, ShuffledPayload
```

## 10.2 必须记录的 CSV

```text
v1213_functional_p3_manifest.csv
v1213_functional_p3_linec.csv
v1213_functional_p3_controls.csv
v1213_functional_p3_failure_table.csv
v1213_functional_p3_subspace_qp.csv
v1213_functional_p3_moment_transport.csv
```

字段：

```text
candidate_id
mechanism_family
dataset
seed
window
functional_batch_size
control_id
base_checkpoint_id
update_norm
functional_norm_ratio
cos_to_adamw
adamw_orthogonal_fraction
holdout_loss_ratio
logit_max_abs_drift
CouplingR2_delta
NoiseSignalLeak_delta
RealSignalReservoirRatio_delta
CEp99_delta
ECE_delta
margin_p10_delta
AUC_proxy_delta
control_gap_vs_best
amortized_overhead_ratio
pareto_pass
fail_reason
```

## 10.3 P3 survivor 标准

P3 candidate survivor 必须满足：

```text
3x3 all-row pass；
or 8/9 pass with bootstrap CI and failure not concentrated in one dataset.
```

Hard gate：

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
holdout\_loss\_ratio \le 1.002,
$$

$$
logit\_max\_abs\_drift \le 0.05,
$$

$$
CEp99\_delta \le 0.05,
$$

$$
control\_gap\_vs\_best \ge 0.005.
$$

No promotion if：

```text
only CouplingR2 improves；
only one dataset/seed passes；
NoOp / Random / AdamWParallel also passes；
update is effectively no-op；
metric pass disappears under repeated probe splits。
```

---

# 11. P4 Short-Run Functional Validation

P4 只在 P3 survivor 出现后运行。

## 11.1 目标

验证 P3 的 functional mechanism 能否转化为连续训练收益，而不是一次性 clone-row artifact。

## 11.2 方法矩阵

```text
B320-AdamW
B320-NoOpMatchedOverhead
B320-RandomMatchedNorm
B320-AdamWParallelDirection
B320-SNR-only
B320-BM2-survivor
B320-BM2-survivor+BM4-transport
B320-BM5-survivor, if any
MLP-AdamW analog, diagnostic
```

设置：

```text
datasets = MNIST, Fashion-MNIST, KMNIST
seeds = 0,1,2
train_size = 1024
val_size = 512
test_size = 512
epochs = 3 baseline, 5 confirm
event_interval = 24,48
event_warmup = after 20% steps
functional_batch_size = 32 or 64
```

## 11.3 记录指标

```text
method
dataset
seed
step
event_count
accepted_event_count
rejected_event_count
fallback_count
acc_delta_vs_B320
AUC_step_delta_vs_B320
AUC_time_delta_vs_B320
ECE_delta_vs_B320
NLL_delta_vs_B320
CEp99_delta_vs_B320
margin_p10_delta_vs_B320
CouplingR2_delta_vs_B320
NoiseSignalLeak_delta_vs_B320
RealSignalReservoirRatio_delta_vs_B320
control_gap_vs_best
amortized_overhead_ratio
step_ratio_q90
memory_ratio_q90
strict_pass
fail_reason
```

## 11.4 P4 pass 标准

必须同时满足：

$$
Acc_{func} \ge Acc_{B320}-0.003,
$$

$$
AUC_{time,func} \le AUC_{time,B320},
$$

$$
ECE_{func} \le ECE_{B320}+0.005,
$$

$$
CEp99_{func} \le CEp99_{B320}+0.05,
$$

$$
CouplingR^2_{func} \ge CouplingR^2_{B320}+0.02,
$$

$$
NoiseSignalLeak_{func} \le NoiseSignalLeak_{B320}-0.01,
$$

$$
RealSignalReservoirRatio_{func} \le RealSignalReservoirRatio_{B320}-0.01,
$$

$$
control\_gap \ge 0.005,
$$

$$
amortized\_overhead \le 1.05.
$$

## 11.5 失败后 Codex 先尝试什么

如果 P3 pass 但 P4 fail：

```text
1. 检查 failure 是 AUC-time、noise、reservoir、tail、overhead 哪一类。
2. 如果 P4 只坏 overhead：做 event amortization，不改 mechanism。
3. 如果 P4 只坏 noise：增加 noise veto 或降低 alpha，提高 beta。
4. 如果 P4 只坏 reservoir：提高 gamma 或加入 BM5 VJP basis。
5. 如果 P4 只坏 AUC：降低 event frequency 或加入 task-safe projection。
6. 如果 P4 全坏：机制是 clone-only artifact，停止该 family。
```

---

# 12. Line D：Classic No-BSpline Family Closure

Line D 不作为 v12.13 主线，但继续并行。B-spline 已冻结：

```text
BSpline.status = RejectedForThisVersion
active_followup = 0
codex_budget = 0
```

Active families：

```text
Rational
Chebyshev
Wavelet
RBF/FastKAN
Fourier
```

## 12.1 当前状态

```text
Rational = TaskBlocked
Chebyshev = TaskBlocked
Wavelet = TaskBlocked
RBF = ExpressionBlocked
Fourier = ExpressionBlocked
BSpline = RejectedForThisVersion
```

## 12.2 Rational plan

Rational 已经有 L3/A4，但 A5 task gate fail。下一步不做 CE tuning，而是做 Line C task-blocked autopsy。

Hypothesis：

$$
H_D1:
\text{Rational 的 blocker 是 coupling/reservoir geometry collapse，而不是 kernel 或 denominator safety。}
$$

记录：

```text
den_p01
den_condition
r_prime_p95
r_double_prime_p95
CouplingR2
NoiseSignalLeak
RealSignalReservoirRatio
CEp99
AUC_time
ECE
```

尝试方向：

```text
1. group rational + signal readout；
2. denominator-safe tangent metric diagnostic；
3. task-neutral output-scale geometry；
4. residual geometry budget，不改 CE；
5. 如果 Line C collapse 不改善，Rational 保持 TaskBlocked。
```

## 12.3 Chebyshev plan

Chebyshev 已 L3/A4 pass 但 A5 fail。优先检查 degree energy 和 global support noise leak。

记录：

```text
degree_energy_k
high_degree_energy_ratio
recurrence_max_abs
CouplingR2
NoiseSignalLeak
RealSignalReservoirRatio
AUC_time
ECE
```

尝试方向：

```text
1. degree-energy damping；
2. inputcross scale cap；
3. K3/K4 low-degree compare；
4. if NoiseSignalLeak high, freeze high-degree branch early；
5. no dataset-specific threshold。
```

## 12.4 Wavelet plan

Wavelet 已从 KernelBlocked 推进到 TaskBlocked。下一步查 local support coverage 是否不均。

记录：

```text
active_support_count
scale_energy
scale_dead_fraction
local_tail_coverage
CEp99
margin_p10
CouplingR2
NoiseSignalLeak
```

尝试方向：

```text
1. hat wavelet K4/K6 local support compare；
2. scale-energy balance；
3. tail-safe local correction；
4. if A5 still fail but Line C improves, keep as geometry diagnostic only。
```

## 12.5 RBF / FastKAN plan

RBF 现在是 ExpressionBlocked。L3 fused efficiency 已有路径，但 A4 不过。

记录：

```text
K_active
active_center_entropy
dead_center_fraction
out_of_grid_fraction
E1/E2/E6/E8 expression deltas
frozen_R2
trainable_R2
```

尝试方向：

```text
1. K_active 2 -> 4 -> 6；
2. fixed uniform centers vs train-stream quantile centers；
3. add linearraw / inputcross low-cost path；
4. if A4 fail persists, keep ExpressionBlocked，不进 A5。
```

## 12.6 Fourier plan

Fourier 仍是 ExpressionBlocked。先不学频率和相位，只做 low-frequency fixed basis + linear/inputcross。

记录：

```text
frequency_count
high_freq_energy_ratio
spectral_entropy
expression_delta
NoiseSignalLeak
CEp99
```

尝试方向：

```text
1. K2/K4 fixed low-frequency；
2. lowfreq + linearraw；
3. lowfreq + inputcross；
4. late-enable high-frequency residual diagnostic；
5. if expression still fail, remain ExpressionBlocked。
```

---

# 13. 必须生成的可视化

v12.13 必须生成以下图；缺图不能算分析完成。

## 13.1 B320 anchor dashboard

```text
fig_A_b320_scorecard.svg
fig_A_b320_auc_seed_matrix.svg
fig_A_b320_efficiency_memory_bar.svg
fig_A_b320_linec_vs_mlp.svg
```

## 13.2 Line C calibration

```text
fig_C_noop_null_distribution.svg
fig_C_metric_std_by_batch_size.svg
fig_C_noise_leak_ci_by_method.svg
fig_C_reservoir_ci_by_method.svg
fig_C_coupling_vs_noise_tradeoff.svg
```

## 13.3 Functional mechanism

```text
fig_B_functional_pareto.svg
fig_B_noise_leak_vs_coupling.svg
fig_B_reservoir_vs_auc_time.svg
fig_B_control_gap_by_candidate.svg
fig_B_p3_to_p4_prediction.svg
fig_B_event_accept_reject_timeline.svg
fig_B_qp_constraint_activity.svg
fig_B_moment_transport_effect.svg
```

## 13.4 Classic family

```text
fig_D_family_status_grid.svg
fig_D_family_efficiency_expression_task.svg
fig_D_rational_linec_collapse.svg
fig_D_cheby_degree_energy.svg
fig_D_wavelet_local_coverage.svg
fig_D_rbf_expression_r2.svg
fig_D_fourier_spectrum_expression.svg
```

---

# 14. 并行执行安排

为了加速，不再串行等待一个 blocker。v12.13 采用四个并行 batch。

## Batch 1：测量可信度与 anchor 复核

```text
A0: B320 exact anchor monitor
C0: Line C calibration / null distribution
D0: No-BSpline family status refresh
```

Batch 1 输出：

```text
v1213_b320_anchor_monitor.csv
v1213_linec_calibration.csv
v1213_linec_null_distribution.csv
v1213_family_status_no_bspline.csv
```

## Batch 2：机制探针并行

```text
B-M1-ext: Multi-window SNR
B-M2: Constrained LineC QP / signal-reservoir projector
B-M3-ext: Branch rebalance with logit LS
B-M4: Moment/state transport
B-M5: Reservoir-targeted VJP correction
D-line: Rational/Cheby/Wavelet/RBF/Fourier repairs
```

Batch 2 输出：

```text
v1213_functional_p3_linec.csv
v1213_functional_p3_controls.csv
v1213_functional_p3_failure_table.csv
v1213_family_efficiency.csv
v1213_family_expression.csv
v1213_family_task_linec.csv
```

## Batch 3：P4 short-run

只对 P3 survivor 运行。

```text
B320 + survivor
B320 + survivor + BM4 transport
B320 + controls
```

Batch 3 输出：

```text
v1213_functional_p4_short_run.csv
v1213_functional_p4_trace.csv
v1213_functional_p4_control_gap.csv
```

## Batch 4：10-seed confirm / external preparation

只有 P4 pass 后打开。

```text
B320 baseline 10-seed
B320 + functional 10-seed
MLP same-param / same-step / same-time controls
classic family survivor if any
```

---

# 15. 最终 route 决策

v12.13 route 不允许含糊。

## R1：B320 anchor regression

```text
B320 strict anchor fail。
停止 functional，先修复 B320 implementation/protocol。
```

## R2：Line C metric not reliable

```text
NoOp/Random false positive 高，Noise/Reservoir CI 不稳定。
停止 functional promotion，先修 metric calibration。
```

## R3：Functional mechanism partial only

```text
存在 1/9 或少量 partial pass，但 3x3 不过。
不能 P4，只能机制诊断。
```

## R4：Coupling-only mechanism

```text
CouplingR2 高，但 NoiseSignalLeak / Reservoir 不改善。
该机制只能作为 coordinate diagnostic，不是 functional value source。
```

## R5：P3 full pass but P4 fail

```text
clone one-step 机制无法转化为 continuous training。
进入 P3-to-P4 autopsy，不扩种子。
```

## R6：Functional local success

```text
P3 + P4 pass，击败 controls，task non-harm，overhead pass。
可以声明 local functional evidence，但不能声明 full success。
```

## R7：Functional full success

```text
10-seed confirm + strong controls + Line C + external fair precheck pass。
可以进入论文主 claim。
```

---

# 16. 本计划的 justification

v12.13 这样设计的理由很直接：

1. **B320 已经解决 base 问题。** 现在继续找 base 或小修 B320，只会拖慢 functional 机制验证。
2. **旧 P3 score 已被证伪。** Spearman / Pearson 都太低，不能继续 promotion。
3. **Line C 不是一个新分数，而是 functional 的价值定义。** 但它必须先校准测量噪声，避免把不可达阈值当成机制失败。
4. **CouplingR2 不能单独代表好几何。** BM3 已经证明 CouplingR2 可以被 coordinate move 轻易刷高；真正 hard target 是 NoiseSignalLeak 和 RealSignalReservoirRatio。
5. **BM1 有局部机制信号。** 这说明不是完全没路，但 coverage 太低，必须做 multi-window / role-wise / CI-gated 扩展。
6. **功能更新必须直接约束 noise/reservoir。** 不能再先造 update 再看指标；BM2/BM5 应该直接把 Line C hard constraints 放进求解过程。
7. **经典基函数保留但不抢主线。** Rational/Cheby/Wavelet 已是 TaskBlocked，RBF/Fourier 是 ExpressionBlocked，B-spline 已冻结。它们继续并行推进，但不能替代 B320 functional 的主科学问题。

最终，这份计划聚焦一个核心判断：

$$
\boxed{
\text{DG-KAN 现在已经不是“能不能有快 base”的问题，}
\text{而是“functional update 是否有独立、可部署、population-safe 几何价值”的问题。}
}
$$

