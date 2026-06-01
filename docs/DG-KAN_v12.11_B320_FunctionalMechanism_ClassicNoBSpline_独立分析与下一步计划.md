# DG-KAN v12.11：B320 Anchor Lock + Functional Mechanism Rebuild + Classic No-BSpline 支线完整计划

> 版本：v12.11 execution plan  
> 基于：v12.10 `B320 Functional ClassicNoBSpline` 执行复盘  
> 核心约束：strict FC-PureKAN；CE-only；no teacher / distillation；no loss modification；no sampler / class weight；no dataset-name branch；no fake / proxy / CPU offload；KAN official path 不使用 PyTorch `loss.backward()` 作为训练更新路径；B-spline 本版冻结，不作为 active family。  
> 公式格式：Typora 友好，只使用 `$...$` 与 `$$...$$`。

---

# 0. 总判断

v12.10 是一个关键分水岭。它不是“没进展”，也不是“functional update 成功”。它真正说明：

$$
\boxed{\text{B320 base 线已经基本打穿；functional update 线仍未打穿。}}
$$

本轮最重要的进展是 B320 strict hardening 通过：

```text
B320_F3_step_ratio_q90 = 0.7938351796804363
B320_F3_memory_ratio_q90 = 0.1295238095238095
B320_mean_delta = 0.022526041666666666
B320_worst_delta = 0.0
B320_near_pass_rate = 1.0
B320_max_ECE_delta = 0.019615232944488525
B320_max_AUC_step = 0.9809975399787746
B320_max_AUC_time = 0.9804242794311487
B320_LineC_nontearing_pass = 1
B320_v1210_strict_hardening_pass = 1
```

这意味着我们不再处在“有没有一个可用 PureKAN base”的阶段。现在已经有了一个可用的 strict FC-PureKAN anchor。下一步不应该把大量资源继续用在 B314/B109/B320 的小超参修补上，而应该围绕 B320 做两件更本质的事：

```text
1. 证明 functional update 在 B320 上有独立、可连续训练、control-resistant 的几何收益。
2. 继续推进非 B-spline 经典 basis family，尤其是已经打开 fused L3 的 family，把它们从 efficiency/expression/task/geometry blocker 推进到明确状态。
```

当前 route 是：

```text
route = R5-B320P4ShortRunFailed
official_functional_success = 0
functional_P3_official_candidate_pass = 1
classic_family_pass_count_excluding_bspline = 0
```

这说明 functional update 已经从“P3 都没过”推进到“P3 有强 diagnostic signal，但 P4 连续训练不成立”。这是进展，但不是终点。

---

# 1. 独立分析：这次到底进展在哪里

## 1.1 B320 已经从 candidate 变成 anchor

B320 的核心价值是它同时满足了以前长期无法同时满足的几件事：

```text
1. F3 efficiency 过：step ratio < 1，memory ratio 极低。
2. task hardening 过：mean delta 为正，worst delta 不负，near pass rate = 1。
3. AUC-step / AUC-time 过：训练轨迹不再只是 final accuracy 好看。
4. ECE 没有超过约定阈值。
5. Line C nontearing pass。
```

所以 v12.10 后，B320 的角色应当改变：

```text
过去：B320 是一个需要继续竞争的 base candidate。
现在：B320 是 v12.11 functional re-entry 的 official anchor。
```

这不代表 B320 已经足以支撑最终论文结论，因为 external fair、更大任务、不同 train-size/batch-size 仍未完成。但对于当前 MNIST/Fashion/KMNIST strict FC-PureKAN protocol，它已经足够作为 functional update 的承载 base。

## 1.2 Functional update：P3 强信号没有转成 P4 成功

P3 gate 给出一个很强的正信号：

```text
best_control = C0-TaskOnlyAdamW
best_control_delta_score = 0.112838347053971
best_functional_update = F14d-OrthDirectBranchDampingOnly
best_functional_delta_score = 0.3833136147673263
control_gap = 0.2704752677133553
CouplingR2_best_control = 0.11466433534666509
CouplingR2_after = 0.33411868278337853
CouplingR2_delta_vs_best_control = 0.21945434743671344
NoiseSignalLeak_delta = -0.021574735641479492
P3 official candidate pass = 1
```

如果只看这个表，很容易误以为 functional 已经成功。但 P4 short-run 把这个结论推翻了。P4 做了多轮修复和 bracket：runtime backtracking、fixed lambda、lambda 强度、符号、branch mode、F15-style blend、event-level tail/noise/reservoir veto。结果全部是：

```text
B320-bestFunctional strict pass = 0/9
official_functional_success = 0
```

这说明 P3 的问题不是“假信号”，而是“局部 cloned diagnostic 信号不能稳定转成训练内连续收益”。这类信号最危险，因为它足够强，会诱导我们继续小修 lambda / event interval / branch mode；但 P4 已经显示这些小修没有打穿 gate。

## 1.3 P4 失败不是单一失败，而是机制失败

P4 runtime backtracking 失败很明显：

```text
pass = 0/9
mean acc_delta = -0.004991319444444444
mean ECE_delta = 0.025381920238335926
mean CouplingR2_delta = 0.36917729919583625
mean NoiseSignalLeak_delta = 0.03409979140592946
mean overhead = 6.380373793551417
mean control_gap = -0.41952276882641354
```

这说明 runtime backtracking 虽然可能提升 coupling，但代价、ECE、noise leak 和 control gap 都不合格。

固定 lambda 修复了成本，但没有修成机制：

```text
lambda 0.03125:
  mean CouplingR2_delta = 0.3323082514010865
  mean NoiseSignalLeak_delta = 0.0010810624808073044
  mean overhead = 0.9397015470003984
  mean control_gap = 0.0911458039600335
  pass = 0/9
```

方向反转或 quad mode 可以局部改善 ECE / control gap，但仍达不到 NoiseSignalLeak 和 AUC gate：

```text
neg-quad lambda 0.03125:
  mean ECE_delta = -0.0008270897799068027
  mean CouplingR2_delta = 0.33389939226294857
  mean NoiseSignalLeak_delta = -0.0014243870973587036
  mean control_gap = 0.2801329136210396
  mean CEp99_delta = 0.05871412489149305
  pass = 0/9
```

更强的 neg-quad 提高 noise release，但 CE tail 和 AUC-time 变坏：

```text
neg-quad lambda 0.0625:
  mean NoiseSignalLeak_delta = -0.0030022936148775946
  mean CEp99_delta = 0.133310423956977
  pass = 0/9
```

F15-style task-safe blend 则几乎变成 no-op：

```text
task-minus-quad010 lambda 0.03125:
  mean CouplingR2_delta = 3.9426649639554924e-05
  mean NoiseSignalLeak_delta = 2.2382785876592e-06
  mean control_gap = 4.812251550096828e-05
  pass = 0/9
```

这给出一个清晰结论：

$$
\boxed{
\text{当前 functional direction 不是强度不对，而是机制不够。}
}
$$

它有三种坏模式：

```text
1. 强一点：CouplingR2 好，但 ECE / CEp99 / NoiseSignalLeak / AUC-time 坏。
2. 弱一点：task-safe，但几何效应塌缩。
3. 加 veto：坏事件被拒绝，最后近似 no-op。
```

这不是 lambda grid 能解决的问题。

## 1.4 Event-level veto 是真实负结果

v12.10 加入 tail/noise/reservoir veto 后，所有 9 个 B320-bestFunctional row 的事件被拒绝或仍然不过 gate。尤其 tail_noise_reservoir reuse run 中：

```text
acc_delta_vs_B320 = 0.0
AUC_time_delta_vs_B320 = 0.8153835439055539
CouplingR2_delta_vs_B320 = 1.0306551399527938e-05
RealSignalReservoirRatio_delta_vs_B320 = 1.3278590308295356e-06
NoiseSignalLeak_delta_vs_B320 = 6.796585188971626e-07
control_gap_vs_best = -0.013803450879180236
accepted_event_count = 0.0
rejected_event_count = 1.0
```

这说明 current event-gated functional 不能靠 safety veto 变成成功。如果安全门足够严格，它就全拒绝；如果放松，它就带来 tail/noise/AUC 风险。

## 1.5 经典 basis 支线：不是没有进展，但没有 FamilyPass

B-spline 已按计划冻结，本轮不能再把它写成新失败。

非 B-spline family 的状态更细：

```text
Chebyshev：L3 efficiency 已经很健康，但 A4 expression 失败。
Fourier：L3 efficiency 已经健康，尤其 B4w memory 很低，但 A4 expression 更差。
RBF/FastKAN：新增 fused_rbf Triton K2/K4 后，L3 efficiency 已打开，但 A4 expression 失败。
Wavelet：新增 fused_hat_wavelet Triton K4 后，L3 efficiency 已打开，但 A4 expression 失败。
Rational：不能把 focused MNIST seed0 的 FamilyPass 当 official；full multi-dataset evidence 仍不是 family success。
```

这其实是好消息和坏消息的混合：

```text
好消息：RBF/Wavelet 从 KernelBlocked 推进到了 ExpressionBlocked。
坏消息：classic family 当前没有任何一个形成 official family success。
```

也就是说，经典 basis 线的 blocker 已经开始从“跑不快”转向“快了但表达/任务/几何不够”。这比完全 KernelBlocked 更接近可研究状态。

---

# 2. 现在卡在哪里

当前 blocker 不再是单纯效率，不再是 base 是否存在，而是：

$$
\boxed{
\text{functional update 的 P3 局部几何信号，不能在 P4 连续训练中保持 task-safe、control-resistant、noise-safe。}
}
$$

它背后的本质问题有四个。

## 2.1 CouplingR2 被单独优化会误导

P4 中多次出现：

```text
CouplingR2_delta 大幅为正，但 NoiseSignalLeak / CEp99 / AUC_time 不过。
```

这说明 CouplingR2 不是单独目标。它只能和以下指标一起解释：

```text
NoiseSignalLeak
RealSignalReservoirRatio
CEp99
margin_p10
ECE
AUC_step / AUC_time
control_gap_vs_best
```

因此 v12.11 不能继续用“提升 CouplingR2”作为 functional update 的主目标。更正确的目标是：

$$
\boxed{
\text{提高 train-probe coupling 的同时，不能把噪声送进 signal channel，不能损伤 tail，不能输给 controls。}
}
$$

## 2.2 P3 评价与 P4 训练存在分布/时序错配

P3 是 cloned one/five-step diagnostic。P4 是训练中的 repeated event。P3 pass 但 P4 fail，说明至少存在一种错配：

```text
1. P3 的 checkpoint/state 与 P4 事件实际触发 state 不同；
2. P3 的一次性 functional displacement 在 repeated application 下累积成 tail/noise 风险；
3. P3 的 score 使用的是局部 train/probe window，不能预测 later-window AUC；
4. P3 的 branch damping 主要改变 output geometry，但没有改善 task descent path；
5. P3 的 control baseline 不够覆盖 P4 中的 runtime event dynamics。
```

v12.11 必须先做 P3-to-P4 causal bridge audit，而不是继续扩展 functional 候选名字。

## 2.3 当前 functional 更像 branch-space damping，不像 population-risk update

F14d 的名字是 `OrthDirectBranchDampingOnly`。它的效果符合这个名字：它能在 branch/output 子空间制造几何变化，但在连续训练中没有证明它释放真实信号、减少噪声泄漏、加快 AUC。

这说明 functional update 应从“branch damping”转向更直接的 population-risk / signal-reservoir 机制：

```text
不是：某个 branch 该 damp 多一点。
而是：哪个训练方向把 real signal 从 reservoir 放进 signal channel，同时不把 noise 放进 signal channel。
```

## 2.4 Classic family 的新 blocker 是 expression/task，不再只是 kernel

RBF/Wavelet 的 fused L3 结果很重要，因为它说明 lower-level kernel 不是不可能：

```text
RBF B2s fused L3 step_ratio = 0.6920597147480739，memory ≈ 1.0048，但 A4 fail。
Wavelet B5h fused L3 step_ratio = 0.648993247415022，memory ≈ 1.0048，但 A4 fail。
```

这意味着 classic family 下一步不能再只说“缺 fused kernel”。对于 RBF/Wavelet，已经有 focused fused kernel；下一步是表达容量和 task-geometry repair，同时保持 L3。对于 Chebyshev/Fourier，也是 expression repair，而不是 timing repair。

---

# 3. 是否在正确道路上

结论：

$$
\boxed{\text{大方向正确，但 v12.11 必须切换问题层级。}}
$$

正确的地方：

```text
1. B320 base 已经被严格 hardening，而不是把 near-pass 包装成 pass。
2. Functional 没有因为 P3 过就宣布成功，而是跑了 P4 short-run。
3. P4 尝试了成本、强度、符号、branch role、event veto，证明不是小修没扫够。
4. B-spline 按计划冻结，没有继续消耗预算。
5. Classic family focused run 没把 MNIST seed0 结果包装成 full family success。
```

需要改变的地方：

```text
1. 不再继续围绕 F14/F15 做 lambda / event_every / branch mode 小修。
2. 不再把 CouplingR2 当成单独 functional target。
3. 不再把 event veto 当成主机制；它只是 safety，不是 value source。
4. Classic family 不能只说 KernelBlocked；对 RBF/Wavelet/Ch/Fourier 要进入 expression/task repair。
```

---

# 4. 离目标还差多远

| 模块 | 当前状态 | 距离目标 | 判断 |
|---|---|---:|---|
| B320 base | strict hardening pass | 近 | 当前已可作为 anchor，但仍需锁协议和更宽确认 |
| Efficiency | B320 step 0.794, memory 0.130 | 近 | base efficiency 已经足够，不是当前主 blocker |
| Line C base geometry | B320 nontearing pass | 近 | 可作为 functional 测试地基 |
| Functional P3 | official candidate pass | 中 | 有强 diagnostic signal |
| Functional P4 | 0/9 pass | 远 | 连续训练未成立，是主 blocker |
| Classic Cheb/Fourier | L3 pass, A4 fail | 中 | 从 efficiency 进入 expression repair |
| Classic RBF/Wavelet | fused L3 focused pass, A4 fail | 中 | 从 KernelBlocked 推到 ExpressionBlocked |
| Rational | task/geometry/coupling 未闭合 | 中远 | 需机制 autopsy，不是 CE tune |
| Next-gen MLP claim | 不能 claim | 远 | 还缺 functional 独立优势和外部 fair |

最短路径：

$$
\boxed{
\text{锁定 B320 base，重建 functional update 机制；}
\text{classic family 并行推进，但不让它抢占 B320 functional 主线。}
}
$$

---

# 5. v12.11 总目标

v12.11 的总目标不是“再找一个 base”，也不是“再调一个 functional lambda”。

总目标是：

$$
\boxed{
\text{在已合格的 B320 strict FC-PureKAN anchor 上，}
\text{证明 functional update 是否能带来连续训练中的独立几何收益。}
}
$$

并且继续并行推进非 B-spline classic family：

$$
\boxed{
\text{对 Rational、Chebyshev、Fourier、RBF/FastKAN、Wavelet 做 family-specific repair，}
\text{每个 family 输出明确 status，不把 focused run 冒充 official success。}
}
$$

v12.11 采用四条并行线：

```text
Line A：B320 anchor lock，不再做小超参修补。
Line B：Functional mechanism rebuild，解决 P3-to-P4 mismatch。
Line C：Manifold-Channel Geometry Diagnostics，作为 functional 与 family 的几何证据。
Line D：Classic No-BSpline family repair，重点从 efficiency 转向 expression/task/geometry。
```

---

# 6. 核心假设

## H-A：B320 已足够作为 functional official anchor

B320 当前已经过 strict hardening。v12.11 只需要确认它不是 artifact 或 wrapper 偶然，而不是继续做 B320 小修。

成立标准：

$$
step\_ratio_{q90}\le 0.85,
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
near\_pass\_rate=1.0,
$$

$$
AUC_{step}\le1.00,
$$

$$
AUC_{time}\le1.00,
$$

$$
ECE_{delta,max}\le0.02,
$$

$$
LineC_{nontearing}=1.
$$

若不满足，Codex 先尝试：

```text
1. 检查 protocol/warmup/timing accounting，不改模型。
2. 检查 B320/B321 candidate id 是否混用。
3. 复跑 exact seeds 与 fixed baseline。
4. 只允许 unlabeled/global scale repair；禁止 dataset-specific branch。
5. 若仍失败，回退 B314/B320 对比 autopsy，不打开 functional official。
```

## H-B：P3-to-P4 mismatch 来自 functional score 与连续训练目标错配

成立信号：

```text
P3 predicted improvement 高，但 P4 actual AUC/noise/tail/control 失败；
P3 的 CouplingR2_delta 与 P4 的 AUC_time_delta / NoiseSignalLeak_delta 低相关；
P4 failure mode 随 event repetition 累积。
```

需要验证的公式：

$$
Corr(Score_{P3}, Gain_{P4})
$$

其中：

$$
Gain_{P4}
=
-w_a\Delta AUC_{time}
-w_n\Delta NoiseSignalLeak
-w_t\Delta CEp99
+w_c\Delta CouplingR^2.
$$

如果相关性低于：

$$
Corr<0.30,
$$

则当前 P3 score 不再作为 promotion score，只保留为 diagnostic。

## H-C：Functional update 必须从 branch damping 改成 signal-reservoir population-risk maintenance

Functional 的新目标不是直接提高 CouplingR2，而是：

$$
\boxed{
\Delta CouplingR^2>0,
\quad
\Delta NoiseSignalLeak<0,
\quad
\Delta RealSignalReservoirRatio<0,
\quad
\Delta AUC_{time}\le0,
\quad
\Delta CEp99\le \epsilon.
}
$$

如果某候选只能提高 CouplingR2，但不能降低 NoiseSignalLeak 或 ReservoirRatio，则不能 official。

## H-D：Classic family 当前 blocker 已经分化

各 family 当前假设：

```text
Chebyshev：kernel 已够快，但 global polynomial basis 的 expression coverage 不够；需要 low-rank pair / local rotation / degree-energy control。
Fourier：kernel 已够快，但 low-frequency basis 对 E1/E6/E8 interaction 不够；不能盲目加高频，否则 noise leak 风险上升。
RBF/FastKAN：fused L3 已能跑快，但 K2/K4 local centers 表达不足；需要 compact capacity repair 而不是回到 dense RBF。
Wavelet：hat K4 fused L3 已快，但只对 E2 composition 有效，对 E1/E6/E8 基本失败；需要 multi-scale/local pair coverage。
Rational：主要是 task geometry / coupling collapse，不是 denominator correctness 或单纯 L3。
```

---

# 7. Line A：B320 Anchor Lock

## 7.1 目标

确认 B320 作为 v12.11 functional anchor 的合法性，并冻结其参数配置。Line A 不是继续优化 B320，而是建立可靠地基。

## 7.2 必跑实验

```text
A0 B320 exact rerun：datasets MNIST/Fashion-MNIST/KMNIST, seeds 0..9。
A1 B320 timing-only rerun：独立测 F3 step/memory，禁止 Line C hook 混入 timing。
A2 B320 Line C rerun：CouplingR2 / NoiseSignalLeak / ReservoirRatio。
A3 B314/B320/B321 contrast：确认 B320 是当前 anchor，而不是 B321 或 B314。
A4 MLP same-param / same-step / same-wallclock baseline rerun。
```

## 7.3 必须记录

`v1211_b320_anchor_lock.csv`：

```text
candidate_id
impl_path
dataset
seed
train_size
val_size
test_size
epochs
batch_size
step_ratio_q90
memory_ratio_q90
forward_ratio_q90
backward_ratio_q90
update_ratio_q90
val_acc_delta_vs_mlp
mean_delta
worst_delta
near_pass
AUC_step_ratio
AUC_time_ratio
ECE_delta
CEp99_delta
margin_p10_delta
CouplingR2
NoiseSignalLeak
RealSignalReservoirRatio
LineC_nontearing_pass
strict_pass
fail_reason
```

## 7.4 可视化

```text
fig_A_b320_task_auc_seed_matrix.svg
fig_A_b320_efficiency_distribution.svg
fig_A_b320_linec_vs_mlp.svg
fig_A_b314_b320_b321_contrast.svg
fig_A_b320_gate_ladder.svg
```

## 7.5 判断

B320 通过后：

```text
anchor_status = B320Locked
functional_official_reentry_allowed = 1
```

若 B320 失败：

```text
anchor_status = AnchorUnstable
functional_official_reentry_allowed = 0
```

失败后 Codex 先做：

```text
1. protocol audit：timing warmup、baseline hash、candidate id。
2. B320 vs B314/B321 row-level diff。
3. 如果只是 ECE/AUC 轻微超线，做 global unlabeled scale bracket。
4. 如果 task/worst/near 失败，回退 base repair，不跑 B-line official。
```

---

# 8. Line B：Functional Mechanism Rebuild

## 8.1 目标

解释并修复：为什么 P3 pass 的 F14d 在 P4 short-run 失败。

Line B 不再做：

```text
lambda 小网格；
event_every_steps 小网格；
branch mode direct/quad/both/negative 继续横扫；
只看 CouplingR2 的 promotion；
只靠 event veto 拦截坏更新。
```

Line B 要做的是：

```text
1. P3-to-P4 causal bridge autopsy；
2. 重新定义 functional value；
3. 设计 signal-reservoir population-risk maintenance；
4. 在 P4 short-run 中击败 controls。
```

## 8.2 B0：P3-to-P4 Causal Bridge Autopsy

### 目的

判断 P3 的 positive 是否真的预测 P4 的连续训练收益。

### 方法

对 B320 checkpoint，在同一 seed/dataset 上生成事件：

```text
E0：P3 cloned one-step
E1：P3 cloned five-step
E2：P4 first event only
E3：P4 repeated event 24-step interval
E4：P4 repeated event with fixed lambda
E5：P4 repeated event with event veto
```

对每个事件保存 pre-state / post-state / prediction：

```text
pre_checkpoint_hash
post_checkpoint_hash
event_step
event_lambda
functional_mode
P3_score
P3_CouplingR2_delta
P3_NoiseSignalLeak_delta
P3_ReservoirRatio_delta
P3_CEp99_delta
P4_actual_AUC_time_delta
P4_actual_CouplingR2_delta
P4_actual_NoiseSignalLeak_delta
P4_actual_CEp99_delta
P4_control_gap
```

### 记录文件

`v1211_p3_p4_bridge_autopsy.csv`

```text
run_id
dataset
seed
event_id
event_step
window_id
candidate_id
functional_id
lambda
mode
p3_score
p3_task_delta
p3_coupling_delta
p3_noise_delta
p3_reservoir_delta
p3_tail_delta
p4_auc_time_delta
p4_auc_step_delta
p4_coupling_delta
p4_noise_delta
p4_reservoir_delta
p4_tail_delta
p4_ece_delta
p4_margin_delta
best_control_id
control_gap
accepted_by_gate
actual_strict_pass
failure_reason
```

### 通过标准

当前 P3 score 可继续用作 promotion 的最低条件：

$$
Corr(P3Score, P4Gain)\ge0.30,
$$

并且：

$$
AUC(P3Score, P4StrictPass)\ge0.70.
$$

如果不过，则当前 P3 score 废弃为 diagnostic，只保留其特征组件。

### 可视化

```text
fig_B0_p3_score_vs_p4_gain.svg
fig_B0_p3_coupling_vs_p4_noise.svg
fig_B0_event_repetition_drift.svg
fig_B0_failure_mode_sankey.svg
fig_B0_controls_by_event.svg
```

## 8.3 B1：Functional Value 重新定义

新的 functional value 不能是单一 coupling 分数。定义：

$$
V_{func}
=
+w_c\Delta CouplingR^2
-w_n\Delta NoiseSignalLeak
-w_r\Delta RealSignalReservoirRatio
-w_a\max(0,\Delta AUC_{time})
-w_t\max(0,\Delta CEp99)
-w_e\max(0,\Delta ECE)
-w_m\max(0,-\Delta MarginP10).
$$

默认权重先不合成 official score，先 report Pareto：

```text
Coupling gain
Noise leak reduction
Reservoir release
AUC non-harm
Tail non-harm
Calibration non-harm
Control gap
```

只有当 Pareto pass 后，才允许计算 aggregate score。

### Pareto pass

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
\Delta AUC_{time}\le0,
$$

$$
\Delta CEp99\le\epsilon_{tail},
$$

$$
control\_gap\ge0.005.
$$

## 8.4 B2：候选机制

### F-M1：Signal-Reservoir Release Update

目标：减少真实信号困在 reservoir 的比例，而不是直接 damp branch。

候选方向：

$$
d_{res-release}
= P_{sig}g_{real} - \alpha P_{sig}g_{noise} - \beta P_{tail}g_{tail-risk}.
$$

实现近似：

```text
1. 用 Line C sketch 得到 signal/reservoir projector。
2. 将 update batch residual 拆成 real residual 与 shuffled/noise residual。
3. 构造只释放 real residual、不提升 noise signal leak 的低秩方向。
4. 对 AdamW task direction 做 orthogonalization，避免只是 AdamWParallel。
```

记录：

```text
real_reservoir_release
noise_signal_leak_delta
cos_with_adamw
cos_with_adamwparallel
orthogonal_fraction
```

### F-M2：Noise-Leak Projected Functional

目标：functional update 必须通过 noise leak projection。

定义可接受方向 $d$：

$$
\Delta NoiseSignalLeak(d)\le -0.01
$$

或至少：

$$
\Delta NoiseSignalLeak(d)\le \Delta NoiseSignalLeak(AdamWParallel)-0.005.
$$

失败后不通过调小 lambda 解决，而是修改方向投影。

### F-M3：Tail-Safe Geometry Maintenance

目标：不能牺牲 CEp99 和 margin。

接受条件：

$$
\Delta CEp99\le0.02|CEp99_{base}|,
$$

$$
\Delta MarginP10\ge-0.005.
$$

如果 direction 提升 CouplingR2 但 CEp99 增加，则该方向进入 tail-risk blacklist。

### F-M4：Function-Preserving Branch Reparameterization

目标：改变内部 branch/basis geometry，但保持 logits 近似不变。

这是对 F14 branch damping 的替代：不是直接改输出，而是在 local linearization 下做近似 null-space reparameterization：

$$
J_{logit}\Delta\theta \approx 0,
$$

同时改善：

$$
\Delta Condition<0,
\quad
\Delta BasisEntropy>0,
\quad
\Delta ReservoirRatio<0.
$$

如果它不能带来 task/AUC 改善，仍可作为 geometry maintenance diagnostic，但不能 official functional success。

### F-M5：AdamW-Orthogonal Signal Residual

目标：保留 P3 中 AdamW-orthogonal 的好处，但显式加入 Line C constraints。

方向：

$$
d_{orth}=d_{func}-\frac{\langle d_{func},d_{adam}\rangle}{\|d_{adam}\|^2+\epsilon}d_{adam}.
$$

接受：

```text
orthogonal_fraction >= 0.60
control_gap_vs_AdamWParallel >= 0.005
```

若 $d_{orth}$ 只提升 CouplingR2，不改善 noise/reservoir/AUC，则停止。

## 8.5 B3：Functional P3 Gate v2

P3 v2 不再只看 best delta-score。

每个 candidate 必须输出：

```text
v1211_functional_p3v2_candidate.csv
```

字段：

```text
functional_id
mechanism_class
dataset
seed
checkpoint_step
lambda
cos_with_task_adamw
orthogonal_fraction
holdout_loss_ratio
CouplingR2_delta
NoiseSignalLeak_delta
RealSignalReservoirRatio_delta
CEp99_delta
ECE_delta
MarginP10_delta
AUC_proxy_delta
control_gap_vs_taskonly
control_gap_vs_adamwparallel
control_gap_vs_random
control_gap_vs_snr
P3v2_pareto_pass
P3v2_fail_reason
```

通过：

$$
holdout\_loss\_ratio\in[0.95,1.05],
$$

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
control\_gap\ge0.005.
$$

## 8.6 B4：P4 Short-Run Official Re-entry

只有 P3v2 pass 的 candidate 才进 P4。P4 不再一次只跑 bestFunctional，而是同时跑：

```text
B320-AdamW
B320-NoOpMatchedOverhead
B320-RandomMatchedNorm
B320-AdamWParallelMaintenance
B320-SNR-only
B320-GeometryOnlyNoSNR
B320-bestFunctional-v2
MLP-AdamW
MLP-analogFunctional
```

设置：

```text
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2
steps = 240 or epochs equivalent
functional_event_every = 24, 48
lambda policy = fixed from P3v2, not re-tuned per dataset
```

记录：

```text
v1211_functional_p4_short_run.csv
```

字段：

```text
method
dataset
seed
acc_delta_vs_B320
AUC_step_delta_vs_B320
AUC_time_delta_vs_B320
ECE_delta_vs_B320
CEp99_delta_vs_B320
margin_p10_delta_vs_B320
CouplingR2_delta_vs_B320
RealSignalReservoirRatio_delta_vs_B320
NoiseSignalLeak_delta_vs_B320
amortized_overhead_ratio
functional_event_count
accepted_event_count
rejected_event_count
control_gap_vs_best
strict_pass
fail_reason
```

P4 official pass：

$$
Acc_{func}\ge Acc_{B320}-0.003,
$$

$$
\Delta AUC_{time}\le0,
$$

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
\Delta CEp99\le\epsilon_{tail},
$$

$$
amortized\_overhead\le1.05,
$$

$$
control\_gap\ge0.005.
$$

Row-level requirement：

```text
strict_pass_rate >= 7/9 for exploratory official
strict_pass_rate = 9/9 for hard official
```

若 P4 fail，根据 fail reason 自动分流：

```text
AUC_time_delta>0:
  不调 dataset-specific schedule；先查事件触发 phase，改为 plateau/global geometry-debt trigger。

CouplingR2_delta<0.02:
  方向几何效应不足；停止该 candidate，回到 mechanism design。

NoiseSignalLeak_delta>-0.01:
  方向没有 noise-safe；加入 signal/noise projection，不调 lambda。

RealSignalReservoirRatio release insufficient:
  说明不是 reservoir release 机制；回到 F-M1。

CEp99 / margin fail:
  加 tail-safe projection；若仍 fail，candidate blacklist。

control_gap<=0:
  被 controls 解释；停止，不继续 short-run。

overhead>1.05:
  只允许工程修复，不改数学方向。
```

## 8.7 可视化

```text
fig_B_p3_to_p4_prediction.svg
fig_B_functional_pareto_by_candidate.svg
fig_B_p4_strict_pass_matrix.svg
fig_B_p4_control_gap_by_dataset_seed.svg
fig_B_noise_leak_vs_coupling.svg
fig_B_reservoir_release_vs_auc.svg
fig_B_tail_risk_vs_coupling.svg
fig_B_event_accept_reject_timeline.svg
```

---

# 9. Line C：Manifold-Channel Geometry Diagnostics

Line C 是 v12.11 的基础设施，不是可选图表。

## 9.1 目标

回答：

```text
1. B320 为什么 base geometry 过，而 functional 事件没过？
2. Functional direction 是否真的把真实信号从 reservoir 推向 signal channel？
3. Classic family 的 expression/task failure 是 signal trapped、noise leak，还是 coverage collapse？
```

## 9.2 指标定义

Train-probe coupling：

$$
\Delta U_B=f_{\theta_{t+\Delta}}(B)-f_{\theta_t}(B),
$$

$$
\Delta U_Q=f_{\theta_{t+\Delta}}(Q)-f_{\theta_t}(Q),
$$

$$
A_t=\arg\min_A\|\Delta U_Q-A\Delta U_B\|_F^2+\lambda\|A\|_F^2,
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
RealSignalReservoirRatio=
\frac{\|P_{res}r_{real}\|^2}{\|r_{real}\|^2+\epsilon}.
$$

噪声泄漏到 signal channel：

$$
NoiseSignalLeak=
\frac{\|P_{sig}r_{noise}\|^2}{\|r_{noise}\|^2+\epsilon}.
$$

## 9.3 记录文件

`v1211_linec_diagnostics.csv`

```text
run_id
candidate_id
method
family
basis_family
functional_id
dataset
seed
step
window_size
CouplingR2
CouplingCorr
RealSignalReservoirRatio
NoiseSignalLeak
KernelDrift
CEp99
CEp99_delta
ECE
ECE_delta
NLL
margin_p10
margin_p10_delta
signal_effective_rank
top_eigen_share
reservoir_fraction
hidden_effective_rank
basis_occupancy_entropy
basis_dead_fraction
step_ratio_q90
memory_ratio_q90
nontearing_pass
fail_reason
```

## 9.4 判断

Base nontearing：

$$
CouplingR^2_{candidate}\ge CouplingR^2_{MLP}-0.02,
$$

$$
NoiseSignalLeak_{candidate}\le NoiseSignalLeak_{MLP}+0.02,
$$

$$
CEp99_{candidate}\le CEp99_{MLP}+\epsilon.
$$

Functional improvement：

$$
CouplingR^2_{func}\ge CouplingR^2_{base}+0.02,
$$

$$
NoiseSignalLeak_{func}\le NoiseSignalLeak_{base}-0.01,
$$

$$
RealSignalReservoirRatio_{func}\le RealSignalReservoirRatio_{base}-0.01.
$$

## 9.5 可视化

```text
fig_C_coupling_predicted_vs_actual.svg
fig_C_coupling_r2_by_method.svg
fig_C_signal_spectrum.svg
fig_C_reservoir_ratio_by_method.svg
fig_C_noise_signal_leak_by_method.svg
fig_C_kernel_drift_vs_coupling.svg
fig_C_noise_leak_vs_ece.svg
fig_C_reservoir_vs_auc_time.svg
fig_C_family_linec_status_matrix.svg
```

---

# 10. Line D：Classic No-BSpline Family Repair

B-spline 本版冻结，不再 active。

```text
BSpline.status = FamilyFrozen_KernelBlocked_RejectedForThisVersion
active_followup = 0
codex_budget = 0
```

Active families：

```text
Rational
Chebyshev
Fourier
RBF/FastKAN
Wavelet
```

## 10.1 D0：共同规则

每个 family 必须输出：

```text
FamilyPass
FamilyNearPass
KernelBlocked
ExpressionBlocked
TaskBlocked
GeometryBlocked
RejectedForThisVersion
```

禁止：

```text
1. 用 MNIST seed0 focused run 替代 full family conclusion。
2. 用 efficiency pass 替代 expression pass。
3. 用 A4 pass 替代 A5 / Line C pass。
4. 用 CE tune / class weight / sampler 修 family。
```

共同 artifact：

```text
v1211_family_manifest.csv
v1211_family_efficiency.csv
v1211_family_gradcheck.csv
v1211_family_expression.csv
v1211_family_task_triage.csv
v1211_family_linec.csv
v1211_family_status.json
v1211_family_failure_table.csv
```

## 10.2 Chebyshev：ExpressionBlocked repair

当前状态：L3 step 很好，但 A4 fail。v12.10 focused rows 里 Chebyshev 多个 candidate step 在 `0.65-0.93`，memory 在 `0.78-1.00`，但 E1/E6/E8 delta 仍为负。

### 假设

Chebyshev 的问题不是 recurrence 慢，而是 global polynomial basis 对 interaction target 覆盖不够。

### 候选

```text
CH-F1 Cheby-K4 current fused path baseline
CH-F2 Cheby-K4 + low-rank paircross R16/R32
CH-F3 Cheby-K4 + local rotation 2-block / 4-block
CH-F4 Cheby-K6 degree-energy controlled
CH-F5 Cheby-K4 + tiny quadratic tail, fused-compatible
```

### 记录指标

```text
degree_energy_ratio
high_degree_energy_ratio
recurrence_max_abs
paircross_rank
E1_delta
E2_delta
E6_delta
E8_delta
A4_pass
L3_step_ratio
memory_ratio
NoiseSignalLeak
CouplingR2
```

### 通过

$$
L3\_step\le1.10,
$$

$$
memory\le1.05,
$$

$$
E1,E6,E8\text{ deltas}\ge-0.02,
$$

$$
NoiseSignalLeak\le NoiseSignalLeak_{MLP}+0.02.
$$

失败后 Codex：

```text
1. 若 L3 过但 E1/E6/E8 失败：增加 low-rank paircross，不增加 degree。
2. 若 high degree energy 爆：降 K 或加 degree damping。
3. 若 NoiseSignalLeak 高：冻结高阶项，只保留 K3/K4。
4. 若 paircross 伤效率：block paircross / rank cut。
```

## 10.3 Fourier：ExpressionBlocked repair

当前状态：L3 已开，但 A4 fail，尤其 E6/E8 很差。B4w memory 低，但 expression 更差。

### 假设

Low-frequency Fourier 太全局，缺 interaction/local cover；直接加高频可能只会增加 noise leak。

### 候选

```text
FO-F1 Fourier K2 fused baseline
FO-F2 Fourier K4 lowfreq fused
FO-F3 Fourier K2 + local envelope / window
FO-F4 Fourier K2 + low-rank pair readout
FO-F5 Fourier K2 + quadratic tail residual
FO-F6 Fourier K2 + frequency-energy damping
```

### 记录

```text
frequency_energy_ratio
high_freq_energy_ratio
phase_drift
spectral_entropy
E1/E2/E6/E8 delta
NoiseSignalLeak
CEp99
AUC_step/time
```

### 通过

$$
L3\_step\le1.10,
$$

$$
E1,E6,E8\ge-0.02,
$$

$$
NoiseSignalLeak\le NoiseSignalLeak_{MLP}+0.02.
$$

失败后 Codex：

```text
1. 不直接加高频到 K8/K16。
2. 先加 local envelope 或 pair readout。
3. 如果 A4 仍失败，Fourier 本版标记 ExpressionBlocked，不进入 A5。
```

## 10.4 RBF/FastKAN：Fused L3 已开后的 expression repair

当前状态：fused_rbf K2/K4 forward/backward 已实现，gradient sanity 过，focused L3 efficiency 过；但 A4 失败。B2s 比 B2r 表达好一些但仍不够。

### 假设

K2/K4 fixed-center RBF 太弱；需要 compact capacity repair，但不能回到 dense basis。

### 候选

```text
RBF-F1 K4 fused baseline
RBF-F2 K6/K8 active centers fused
RBF-F3 quantile centers fixed from train stream
RBF-F4 center bank + low-rank readout
RBF-F5 RBF + identity/quadratic residual
RBF-F6 RBF pair-lite interaction, rank R16/R32
```

### 记录

```text
K_active
center_entropy
dead_center_fraction
out_of_grid_fraction
exp_count_per_sample
E1/E2/E6/E8 delta
L3_step_ratio
memory_ratio
CouplingR2
NoiseSignalLeak
```

### 通过

$$
L3\_step\le1.10,
$$

$$
memory\le1.05,
$$

$$
A4=1,
$$

$$
dead\_center\_fraction\le0.30.
$$

失败后 Codex：

```text
1. 若 K4 A4 fail：试 K6/K8 active，不用 dense K。
2. 若 memory > 1.05：改 center tiling / recompute，不存 [B,D,K]。
3. 若 E1/E6 仍 fail：加 low-rank pair-lite。
4. 若 NoiseSignalLeak 高：加 center occupancy regular diagnostic，不改 loss。
```

## 10.5 Wavelet：Fused L3 已开后的 expression repair

当前状态：fused_hat_wavelet K4 L3 step 约 `0.649`，但 A4 失败。它对 E2 composition 较好，E1/E6/E8 几乎失败。

### 假设

single-scale hat wavelet 是 local cover，但缺少跨尺度和 interaction coverage。

### 候选

```text
WAV-F1 Hat K4 fused baseline
WAV-F2 Hat K4 two-scale
WAV-F3 Hat K4 + local pair-lite
WAV-F4 Triangle + quadratic tail
WAV-F5 Haar/step diagnostic only
WAV-F6 Hat K4 + scale-energy balancing
```

### 记录

```text
active_support_count
scale_entropy
scale_dead_fraction
local_tail_coverage
E1/E2/E6/E8 delta
CEp99
margin_p10
NoiseSignalLeak
CouplingR2
```

### 通过

$$
L3\_step\le1.10,
$$

$$
A4=1,
$$

$$
CEp99\le CEp99_{MLP}+\epsilon.
$$

失败后 Codex：

```text
1. 若 E2 好但 E1/E6/E8 坏：加 local pair-lite，而不是单纯加 scale。
2. 若 scale entropy 低：balanced fixed scales。
3. 若 L3 fail：回到 K4 single-scale，不进 A4。
4. 若 task tail 坏：冻结高尺度，保留 low-scale local cover。
```

## 10.6 Rational：Task/GeometryBlocked repair

当前状态：Rational 不能从 focused MNIST seed0 推 official success。已有 evidence 显示 Rational 仍有 task geometry / coupling collapse 问题。

### 假设

Rational 的 blocker 不是 denominator safety 本身，而是 rational group 的 signal-channel geometry 不健康：它可能 A4/L3 局部过，但 A5 / Line C 失败。

### 候选

```text
RAT-F1 B7kc/B7lp/B7lz style baseline
RAT-F2 pairNorm + low hidden residual, bounded
RAT-F3 group rational + tangent metric diagnostic
RAT-F4 output-scale geometry, no CE tune
RAT-F5 rational + B320-style direct branch scaffold
RAT-F6 coupling-preserving rational readout
```

### 记录

```text
den_p01
den_condition
r_prime_p95
r_double_prime_p95
group_function_diversity
group_dead_fraction
CouplingR2
NoiseSignalLeak
RealSignalReservoirRatio
CEp99
ECE
AUC_step/time
```

### 通过

$$
L3\_step\le1.10,
$$

$$
A4=1,
$$

$$
A5=1,
$$

$$
CouplingR^2\ge CouplingR^2_{MLP}-0.02,
$$

$$
NoiseSignalLeak\le NoiseSignalLeak_{MLP}+0.02.
$$

失败后 Codex：

```text
1. 若 denominator 不安全：改 safe denominator，不调 CE。
2. 若 L3 fail：Horner fused + group sharing，不加 full-edge rational。
3. 若 A4 fail：增加 groups 或 small residual，不先提高 degree。
4. 若 A5 fail 且 Line C collapse：做 signal-channel autopsy，停止 CE/ECE tune。
5. 若 task mean 过但 AUC fail：看 reservoir ratio 和 CouplingR2，不调 dataset-specific schedule。
```

---

# 11. 并行执行计划

## 11.1 Batch 0：统一 artifact / no-fake / baseline lock

```text
Owner A：B320 anchor rerun。
Owner B：P3/P4 bridge autopsy scaffold。
Owner C：Line C common library。
Owner D：Classic family manifest without B-spline。
```

输出：

```text
v1211_run_manifest.json
v1211_provenance_audit.csv
v1211_hash_manifest.json
```

## 11.2 Batch 1：B320 anchor + P3/P4 bridge

并行跑：

```text
A0-A4：B320 exact + timing + LineC。
B0：P3-to-P4 bridge autopsy。
C0：Line C metric validation。
```

如果 B320 anchor 不过，B-line 只保留 diagnostic，不进入 official。

## 11.3 Batch 2：Functional mechanism candidates

并行实现：

```text
F-M1 reservoir release
F-M2 noise-leak projected
F-M3 tail-safe geometry
F-M4 function-preserving reparameterization
F-M5 AdamW-orthogonal signal residual
```

每个候选先跑 P3v2，不直接 P4。

## 11.4 Batch 3：P4 short-run

只让 P3v2 pareto pass 的候选进 P4。并行 datasets/seeds：

```text
MNIST seed 0,1,2
Fashion-MNIST seed 0,1,2
KMNIST seed 0,1,2
```

禁止 dataset-specific branch。

## 11.5 Batch 4：Classic family focused repair

并行跑：

```text
Chebyshev expression repair
Fourier expression repair
RBF/FastKAN fused-L3 expression repair
Wavelet fused-L3 expression repair
Rational task/LineC repair
```

每个 family 都输出 status。Focused run 不能覆盖 official family status。

## 11.6 Batch 5：Survivor confirmation

只有满足以下条件才进入：

```text
Functional P4 strict pass >= 7/9；或
Classic family A1/A4/LineC near-pass 且 A5 具备明确正信号。
```

---

# 12. 总 artifact contract

v12.11 必须落盘：

```text
v1211_route_decision.json
v1211_b320_anchor_lock.csv
v1211_b320_efficiency_profile.csv
v1211_linec_diagnostics.csv
v1211_p3_p4_bridge_autopsy.csv
v1211_functional_p3v2_candidate.csv
v1211_functional_p4_short_run.csv
v1211_functional_failure_table.csv
v1211_family_manifest.csv
v1211_family_efficiency.csv
v1211_family_expression.csv
v1211_family_task_triage.csv
v1211_family_linec.csv
v1211_family_status.json
v1211_provenance_audit.csv
v1211_hash_manifest.json
```

---

# 13. 必须生成的可视化

```text
fig_v1211_gate_ladder.svg
fig_A_b320_anchor_scorecard.svg
fig_A_b320_auc_seed_matrix.svg
fig_B_p3_to_p4_prediction.svg
fig_B_functional_pareto.svg
fig_B_p4_fail_reason_heatmap.svg
fig_B_noise_leak_vs_coupling.svg
fig_B_event_accept_reject_timeline.svg
fig_C_signal_reservoir_spectrum.svg
fig_C_reservoir_vs_auc_time.svg
fig_C_noise_leak_vs_tail.svg
fig_D_family_status_matrix.svg
fig_D_family_efficiency_expression_pareto.svg
fig_D_family_bottleneck_waterfall.svg
fig_D_family_linec_radar.svg
```

---

# 14. v12.11 route decision

## Case 1：B320 anchor unstable

```text
route = R1-B320AnchorUnstable
next = base protocol/hardening repair
functional_official = 0
classic_family = continue focused only
```

## Case 2：B320 locked，但 P3-to-P4 score 不可预测

```text
route = R2-P3ScoreNotPredictive
next = discard current P3 score; rebuild functional value from Line C Pareto
functional_official = 0
```

## Case 3：P3v2 有 candidate，但 P4 fail

```text
route = R3-FunctionalP4MechanismFail
next = autopsy by fail reason; no lambda small-grid
functional_official = 0
```

## Case 4：P4 short-run pass

```text
route = R4-FunctionalShortRunPass
next = 10-seed functional confirmation
functional_official = provisional
```

## Case 5：Classic family expression/task survivor

```text
route = R5-ClassicFamilyNearPass
next = family confirm + Line C + optional basis-aware functional diagnostic
```

## Case 6：No functional and no family survivor

```text
route = R6-B320BaseOnlyFunctionalStillBlocked
next = publish internal base milestone; rethink functional mechanism, not base
```

---

# 15. 最终建议

v12.11 的执行优先级：

```text
Priority 1：锁定 B320 anchor。
Priority 2：做 P3-to-P4 causal bridge autopsy。
Priority 3：用 Line C Pareto 重建 functional value。
Priority 4：只让真正 signal/noise/reservoir-safe 的 functional 候选进 P4。
Priority 5：并行推进 Cheb/Fourier/RBF/Wavelet/Rational，但不让 family 支线抢占 B320 functional 主线。
```

最重要的纪律：

```text
1. 不把 P3 pass 写成 functional success。
2. 不再围绕 F14/F15 小修 lambda。
3. 不用 CouplingR2 单指标 promotion。
4. 不按数据集调参。
5. 不恢复 B-spline active 支线。
6. 不把 focused family run 写成 official family pass。
```

一句话总结：

$$
\boxed{
\text{v12.10 已经把 base 问题基本解决，}
\text{v12.11 的主战场是 functional mechanism，}
\text{而不是再造 base 或微调事件强度。}
}
$$
