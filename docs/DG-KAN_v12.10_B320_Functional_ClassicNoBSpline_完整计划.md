# DG-KAN v12.10：B320 Anchor Hardening + Functional Official Re-entry + 经典基函数支线（排除 B-spline）完整计划

> 版本：v12.10 revised execution plan
> 核心修正：B-spline 从 active classic family 中移除，状态固定为 `FamilyFrozen_KernelBlocked_RejectedForThisVersion`。
> 当前主线：B320/FHQ strict efficient base anchor hardening。
> 并行主线：Functional update official re-entry、Manifold-Channel Geometry Diagnostics、经典基函数剩余 family 高效率化。
> 公式格式：Typora 友好，只使用 `$...$` 与 `$$...$$`。
> 硬约束：CE-only；no teacher；no distillation；no loss modification；no sampler / class weight；no dataset-name branch；no fake / proxy / CPU offload；KAN official path 不使用 PyTorch `loss.backward()` graph；functional success 必须击败 strong controls。

---

# 0. 本版计划的核心判断

v12.10 不再把目标写成“寻找至少一个 MLP-like efficient base”。这个表述已经不准确，因为我们现在已经有当前最强 anchor：`B320`。因此 v12.10 的目标不是从零找 base，而是：

$$
\boxed{
\text{锁定 B320 作为 strict FC-PureKAN efficient anchor，}
\text{并在该 base 上证明 functional update 的独立几何价值。}
}
$$

同时，经典基函数支线仍然保留，但 **B-spline 排除出 active 路线**。原因不是 B-spline 作为数学基函数永远没有意义，而是当前 DG-KAN 的证据显示 B-spline 长期停在 kernel / efficiency blocker，尚未进入 task、geometry、functional 层面的科学问题。继续投入会把资源拖回高成本 spline fused-kernel 工程，而不是推进当前最关键的 functional-update 证明。

因此 v12.10 的路线是四线并行：

```text
Line A:
  B320 base anchor hardening。
  目标：确认 B320 在更严格协议下仍然是可锁定的 efficient PureKAN base。

Line C:
  Manifold-Channel Geometry Diagnostics。
  目标：确认 B320 和 functional update 的 train-probe coupling / signal-reservoir / noise leakage 健康。

Line B:
  Functional update official re-entry。
  目标：在 B320 上证明 functional update 相对 AdamW / strong controls 的独立几何收益。

Line D:
  经典基函数剩余 family 高效率化。
  Active: Rational, Chebyshev, Fourier, RBF/FastKAN, Wavelet。
  Frozen: B-spline。
```

这四条线的关系不是并列打榜，而是分工明确：

```text
Line A 提供当前最可靠 base anchor；
Line C 给出“几何是否真的健康”的证据；
Line B 证明 functional update 是否真正有独立贡献；
Line D 防止我们因为 B320 成功而过早放弃其他有潜力的经典 basis。
```

---

# 1. 当前事实锚点

## 1.1 B320 是当前 strict-efficient anchor，而不是普通 near-pass

当前 B320 的关键事实：

```text
B320 10-seed strict task/ECE/AUC:
  rows = 30
  mean_delta = 0.022526041666666666
  worst_delta = 0.0
  near_rate = 1.0
  max_ECE_delta = 0.01961517333984375
  max_AUC_step = 0.9809978238164607
  strict_fail_count = 0

B320 F3 efficiency:
  step_ratio_q90 = 0.9597426809570575
  memory_ratio_q90 = 0.1295238095238095
  backward_ratio_q90 = 1.189502923475249
  update_ratio_q90 = 0.5358158059798208
  official_efficiency_pass = 1

B320-only Line C:
  MLP CouplingR2 = 0.21133869467893396
  B320 CouplingR2 = 0.23002676858920135
  B320 NoiseSignalLeak = 0.04764750599861145
```

这说明 B320 已经不是 B109/B314 那种“接近但仍有 strict AUC slice fail”的状态。B320 的意义是：

$$
\boxed{
\text{PureKAN base 线第一次从 near-pass 推进到可 harden 的 strict-efficient anchor。}
}
$$

但 B320 还需要 hardening，原因有三点：

```text
1. full runner 中 B321 / selected candidate 曾出现 LineCGeometryCollapse，需要 B320 exact 与 B321 exact 分开确认；
2. 当前 train_size / epochs / batch protocol 仍较小，需要扩展 stress protocol；
3. functional official re-entry 还没有成功，B320 只是 base anchor，不是项目终点。
```

## 1.2 Functional update 仍未成功

当前状态：

```text
functional_open = 1
functional official success = 0
best observed B320 branch-damping gap = -0.00014220178469248612
corrected delta-score gap in B320 runs remains negative
```

这说明 functional update 还没有证明独立价值。现在不能说：

```text
B320 + functional 已经比 B320 + AdamW 更好。
```

只能说：

```text
B320 已经提供了 functional official re-entry 的合格承载 base；
functional update 需要重新设计为 control-resistant、signal-channel-aware 的几何维护机制。
```

## 1.3 经典基函数支线：B-spline 冻结，其余保留

当前 classic family 状态：

```text
BSpline:
  status = KernelBlocked
  candidate = B6r
  L3 manual step = 1.50460224850239
  memory = 1.4452380952380952
  blocker = 缺 family-specific fused backward/update kernel

RBF:
  status = KernelBlocked
  candidate = B2r
  L3 manual step = 1.4674100186743748
  memory = 1.6147619047619048

Wavelet:
  status = KernelBlocked
  candidate = B5h
  L3 manual step = 2.740506078472147
  memory = 3.6223809523809525

Chebyshev:
  status = KernelBlocked / ExpressionBlocked depending candidate
  has fused L3 measured paths, but expression gate fails or step slightly exceeds gate

Fourier:
  status = ExpressionBlocked
  fused L3 step around 1.128-1.225 in low-frequency variants, but A4 expression gate fails

Rational:
  status = TaskBlocked / GeometryBlocked
  several variants open L3 and A4, but A5 and Line C fail
```

因此 v12.10 的 classic branch 改为：

```text
Active:
  Rational
  Chebyshev
  Fourier
  RBF/FastKAN
  Wavelet, low priority only

Frozen:
  B-spline
```

B-spline 的状态写死为：

```text
BSpline.status = FamilyFrozen_KernelBlocked_RejectedForThisVersion
BSpline.active_followup = 0
BSpline.codex_budget = 0
```

恢复 B-spline 的唯一条件是：外部或后续已经有一个可直接接入的真实 fused forward+backward/update spline kernel，并且它在我们的 protocol 下先通过：

$$
T_{step}/T_{MLP}\le 1.10,
$$

$$
M_{step}/M_{MLP}\le 0.80.
$$

否则 B-spline 不再消耗 v12.10 Codex 时间。

---

# 2. v12.10 总目标

v12.10 的总目标是：

$$
\boxed{
\text{在 B320 strict efficient PureKAN anchor 上，}
\text{完成 base hardening，并重新打开 functional update 的正式因果验证。}
}
$$

更具体地说，v12.10 要回答四个问题。

## Q1：B320 是否可以锁定为 official base anchor？

判断标准不是单次 10-seed pass，而是在更宽协议下仍满足：

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
near\_rate\ge 1.0,
$$

$$
\max(AUC_{step}, AUC_{time})\le 1.00,
$$

$$
ECE_{max\_delta}\le 0.02.
$$

同时 Line C 必须 nontearing：

$$
CouplingR^2_{B320}\ge CouplingR^2_{MLP}-0.02,
$$

$$
NoiseSignalLeak_{B320}\le NoiseSignalLeak_{MLP}+0.02.
$$

## Q2：B320 的优势来自哪里？

需要拆分：

```text
1. architecture / primitive advantage：B320 vs same-param MLP；
2. system advantage：B320 F3 workspace vs MLP wall-clock / memory；
3. geometry advantage：B320 vs MLP Line C；
4. functional advantage：B320 + functional vs B320 + AdamW / controls。
```

## Q3：Functional update 是否能在 B320 上产生独立收益？

Functional official success 必须满足：

$$
Acc_{func}\ge Acc_{B320}-0.003,
$$

$$
AUCtime_{func}\le AUCtime_{B320},
$$

$$
CouplingR^2_{func}\ge CouplingR^2_{B320}+0.02,
$$

$$
NoiseSignalLeak_{func}\le NoiseSignalLeak_{B320}-0.02,
$$

并且击败：

```text
NoOpMatchedOverhead
RandomMatchedNorm
AdamWParallelDirection
SNR-only
GeometryOnlyNoSNR
ShuffledPayload/Event
MLPAnalogGeometryMaintenance
```

Official control gap 要求：

$$
control\_gap\ge 0.005
$$

或 bootstrap CI lower bound：

$$
CI_{95,lower}(control\_gap)\ge 0.
$$

## Q4：经典 basis 支线还有没有值得保留的技术路线？

排除 B-spline 后，剩余 family 的目标不是替代 B320 主线，而是输出明确状态：

$$
Status(f)\in\{FamilyPass, FamilyNearPass, KernelBlocked, ExpressionBlocked, TaskBlocked, GeometryBlocked, RejectedForThisVersion\}.
$$

每个 family 必须有根因，不允许笼统写成“basis failed”。

---

# 3. 核心假设

## H-A：B320 可以被 harden 成 official strict FC-PureKAN base

B320 已经在 10-seed strict task/ECE/AUC 中无 fail，并且 F3 efficiency 过。假设：

$$
\boxed{
\text{B320 的 pass 不是 protocol 偶然，而是一个稳定的 strict-efficient base。}
}
$$

需要通过扩大协议验证：

```text
train_size = 1024, 2048, 4096
batch_size = 128, 256
seeds = 0..9 first, then selected 0..19 stress if needed
datasets = MNIST, Fashion-MNIST, KMNIST
same-param MLP / same-step MLP / same-wall-clock MLP controls
```

如果 B320 在更大 train_size 下失效，不能按数据集调参。必须先判断：

```text
AUC-step fail:
  primitive / signal trajectory 问题。

AUC-time fail but AUC-step pass:
  kernel / timing / workspace issue。

Line C fail:
  train-probe coupling / noise leakage / reservoir 问题。
```

## H-C：Line C 可以解释 B320 和 functional 的几何状态

假设：

$$
\boxed{
\text{真正好的几何不是 kernel 不漂移，}
\text{而是 train motion 能预测 probe motion，真实信号进入 signal channel，噪声不泄漏。}
}
$$

因此每个 B320 hardening run 和 functional candidate 都必须记录：

```text
CouplingR2
CouplingCorr
RealSignalReservoirRatio
NoiseSignalLeak
KernelDrift
CEp99
ECE
margin_p10
```

## H-B1：Functional 失败不是因为 base 不够，而是因为方向缺少 control-resistant complementarity

过去 functional candidates 多数 task-safe，但没有击败 strong controls。现在 B320 base 合格后，functional 的 blocker 更清楚：

$$
\boxed{
\text{functional direction 必须改善 AdamWParallel / SNR-only 无法改善的几何子空间。}
}
$$

因此 v12.10 不继续扫 residual weight 小网格，而测试四类机制：

```text
F1 AdamW-orthogonal signal residual。
F2 reservoir-release update。
F3 noise-leak vetoed geometry maintenance。
F4 tail-safe signal-channel maintenance。
```

## H-D1：Rational 的 blocker 是 task geometry / coupling collapse，不是 denominator unsafe

Rational 已经有 L3/A4 可行候选，且 denominator safety 诊断未显示分母接近 0。真正 blocker 是 A5、ECE/AUC、Line C coupling collapse。假设：

$$
\boxed{
\text{Rational 需要新的 signal-channel task geometry，}
\text{而不是继续 CE tune 或 pair-readout rank/gain 小修。}
}
$$

## H-D2：Chebyshev / Fourier 当前主要是 ExpressionBlocked

Chebyshev / Fourier 已有 fused L3 路线或接近 fused L3 路线，但 A4 expression fail。假设：

$$
\boxed{
\text{低成本 orthogonal / spectral basis 可以保留，}
\text{但必须补 expression capacity 且不能牺牲 L3 efficiency。}
}
$$

## H-D3：RBF/FastKAN / Wavelet 只有在 compact local fused path 下才值得保留

RBF 和 Wavelet 当前 KernelBlocked。RBF 只允许 FastKAN/compact-local；Wavelet 只允许 hat/triangle-local。Morlet/MexicanHat、dense RBF、dense wavelet 暂停。

## H-D4：B-spline 在 v12.10 冻结

假设：

$$
\boxed{
\text{B-spline 当前投入产出比低于 B320 functional / Rational / Cheby-Fourier 修复。}
}
$$

因此 B-spline 不进入 Codex active task queue。

---

# 4. Phase P0：Route freeze、artifact contract、B-spline freeze

## 4.1 目标

在所有实验开始前冻结路线，避免 Codex 自动恢复 B-spline 或继续旧的 spline kernel repair。

## 4.2 必须修改 / 输出

新增或更新：

```text
v1210_route_config.json
v1210_family_policy.json
v1210_provenance_audit.csv
```

`v1210_family_policy.json` 必须包含：

```json
{
  "active_families": ["Rational", "Chebyshev", "Fourier", "RBF_FastKAN", "Wavelet"],
  "frozen_families": {
    "BSpline": {
      "status": "FamilyFrozen_KernelBlocked_RejectedForThisVersion",
      "active_followup": 0,
      "codex_budget": 0,
      "reactivation_condition": "external_or_existing_fused_forward_backward_update_kernel_passes_L3"
    }
  }
}
```

## 4.3 P0 gate

P0 通过条件：

```text
B-spline candidates not scheduled = 1
no dataset-name branch = 1
no teacher/distillation/loss/sampler/class_weight = 1
no fake/proxy/cpu = 1
B320 exact candidate id resolved = 1
B321 exact candidate id resolved = 1
strong controls manifest complete = 1
```

如果 P0 不过，不允许进入 P1。

---

# 5. Phase P1：B320 strict base hardening

## 5.1 目标

确认 B320 是否能成为 official base anchor。这里不是继续小修 B314/B315/B109，而是：

```text
B320 exact
B321 exact
B314 reference
B109 reference
MLP same-param
MLP same-step / same-wall-clock
```

在同一协议下跑 full hardening。

## 5.2 实验设计

### P1-A：Exact 10-seed replay

```text
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0..9
train_size = 1024
val_size = 512
test_size = 512
epochs = 3
batch_size = 128
task_compile_warmup_steps = 12
```

方法：

```text
MLP-same-param-AdamW
MLP-same-step-AdamW
B109b-reference
B314b-reference
B320b-exact
B321b-exact
```

### P1-B：Protocol stress replay

只对 P1-A pass 的 candidates 跑：

```text
train_size = 2048, 4096
batch_size = 128, 256
epochs = 3, 5
seeds = 0..4 initially; if pass, seeds = 0..9
```

### P1-C：Timing truth / workspace replay

每个 candidate 必须记录：

```text
F1 eager / reference
F2 learnableP
F3 workspace learnableP
F4 fixed/semi-fixed workspace, if applicable
```

不要把 Line C hooks 或 diagnostics 混进 architecture timing。

## 5.3 必须记录的字段

`v1210_b320_base_hardening.csv`：

```text
run_id
candidate_id
method_family
dataset
seed
train_size
val_size
test_size
epochs
batch_size
impl_path
step_ratio_q90
memory_ratio_q90
forward_ratio_q90
backward_ratio_q90
update_ratio_q90
compile_warmup_steps
acc_delta
val_loss_delta
AUC_step_ratio
AUC_time_ratio
ECE_delta
NLL_delta
CEp99_delta
margin_p10_delta
near_pass
strict_fail_reason
no_fake
```

## 5.4 P1 official gate

B320 official base hardening pass：

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
near\_rate=1.0,
$$

$$
\max(AUC_{step},AUC_{time})\le 1.00,
$$

$$
ECE_{max\_delta}\le 0.02.
$$

如果 B320 exact 过、B321 不过，则 B320 locked。
如果 B321 exact 过但 B320 Line C 更好，则 B320 remains primary, B321 remains task-balanced backup。
如果二者都过，进入 P2 Line C 比较后再决定 anchor。
如果二者都不过，不得按 dataset 调参；进入 P1-failure autopsy。

## 5.5 P1 失败后的 Codex 自动尝试方向

### Case P1-F1：AUC-time fail but AUC-step pass

说明主要是 system / timing 问题。Codex 先尝试：

```text
1. 分离 architecture timing 与 diagnostics hook timing；
2. 增加 warmup steps，确认 no late compile；
3. 检查 F3 workspace allocation / update path；
4. 尝试 CUDA graph fixed-shape capture；
5. 不改模型结构，不调 dataset-specific 参数。
```

### Case P1-F2：AUC-step fail

说明是 task trajectory 问题。Codex 先尝试：

```text
1. 输出 per-step val loss trace；
2. 计算 early/mid/late loss slope；
3. 检查 margin_p10 / CEp99 是否同步恶化；
4. 跑 Line C window at failing step；
5. 只允许 dataset-agnostic structural diagnosis，不允许 Fashion/KMNIST-specific rule。
```

### Case P1-F3：ECE / CEp99 fail

Codex 先尝试：

```text
1. 检查 logit norm trajectory；
2. 检查 margin distribution；
3. 检查 NoiseSignalLeak；
4. 不做 post-hoc calibration；
5. 不改 CE loss / temperature by dataset。
```

### Case P1-F4：step ratio fail

Codex 先尝试：

```text
1. profiler component breakdown；
2. F3 workspace update path audit；
3. temporary allocation audit；
4. kernel count / synchronization count；
5. 不用 PyTorch loss.backward 作为 official path。
```

## 5.6 必须可视化

```text
fig_b320_accuracy_delta_by_dataset_seed.svg
fig_b320_auc_step_time_by_dataset_seed.svg
fig_b320_ece_cep99_margin_heatmap.svg
fig_b320_efficiency_stack.svg
fig_b320_loss_vs_step.svg
fig_b320_loss_vs_time.svg
fig_b320_protocol_stress_pareto.svg
```

---

# 6. Phase P2：B320 / B321 Line C geometry hardening

## 6.1 目标

确认 B320 的几何健康不是偶然，也排除 B321 selected-run collapse 的混淆。

Line C 不是 optional appendix。它是判断 base 是否可以承载 functional update 的必要诊断。

## 6.2 核心定义

从训练流中取 update batch $B$ 和 probe batch $Q$，在窗口 $[t,t+\Delta]$ 内记录：

$$
\Delta U_B=f_{\theta_{t+\Delta}}(B)-f_{\theta_t}(B),
$$

$$
\Delta U_Q=f_{\theta_{t+\Delta}}(Q)-f_{\theta_t}(Q).
$$

拟合 ridge transfer：

$$
A_t=\arg\min_A \|\Delta U_Q-A\Delta U_B\|_F^2+\lambda\|A\|_F^2.
$$

Coupling：

$$
CouplingR^2=1-\frac{\|\Delta U_Q-A_t\Delta U_B\|_F^2}{\|\Delta U_Q\|_F^2+\epsilon}.
$$

用 projected gradient / logit-Jacobian sketch 构造：

$$
\hat K_{BB}=\Phi\Phi^\top,
$$

窗口累积：

$$
\hat W_B=\sum_{\tau=t}^{t+\Delta}\hat K_{BB}(\tau).
$$

由 $\hat W_B$ 得到 signal projector $P_{sig}$ 和 reservoir projector $P_{res}$。

真实信号困在 reservoir：

$$
RealSignalReservoirRatio=\frac{\|P_{res}r_{real}\|^2}{\|r_{real}\|^2+\epsilon}.
$$

噪声泄漏：

$$
NoiseSignalLeak=\frac{\|P_{sig}r_{noise}\|^2}{\|r_{noise}\|^2+\epsilon}.
$$

## 6.3 实验设计

对以下方法跑 Line C：

```text
MLP-same-param
B314-reference
B320-exact
B321-exact
B320 under protocol stress
Functional candidates from P3/P4
Strong controls
Active classic family best candidates, excluding B-spline
```

窗口：

```text
steps = early, mid, late
window_size = 1, 5, 20
probe_batch_size = 128 or 256
sketch_dim = 32, 64
ridge_lambda = 1e-3, 1e-2
```

## 6.4 必须记录

`v1210_linec_geometry.csv`：

```text
run_id
candidate_id
method
basis_family
dataset
seed
step
window_size
probe_batch_size
sketch_dim
ridge_lambda
CouplingR2
CouplingCorr
coupling_residual_norm
KernelDrift
signal_effective_rank
signal_mass_topk
reservoir_fraction
top_eigen_share
dissipation_condition
RealSignalReservoirRatio
NoiseSignalLeak
CEp99
ECE
NLL
margin_p10
perturb_logit_drift_p95
LineC_interpretation
```

## 6.5 Line C pass / fail 规则

Base nontearing pass：

$$
CouplingR^2_{KAN}\ge CouplingR^2_{MLP}-0.02,
$$

$$
NoiseSignalLeak_{KAN}\le NoiseSignalLeak_{MLP}+0.02,
$$

$$
CEp99_{KAN}\le CEp99_{MLP}+\epsilon_{tail}.
$$

Interpretation：

```text
good_feature_learning:
  KernelDrift may be high, but CouplingR2 high and NoiseSignalLeak low.

bad_tearing:
  CouplingR2 low, NoiseSignalLeak high, CEp99/ECE bad.

lazy_underfit:
  KernelDrift low, RealSignalReservoirRatio high, task loss not decreasing.

coupling_collapse:
  CouplingR2 significantly below MLP and task/AUC unstable.
```

## 6.6 失败后的 Codex 自动尝试方向

如果 `CouplingR2` 不稳定：

```text
1. 增大 probe batch；
2. classwise center logits；
3. 增加 ridge_lambda；
4. 缩短 window；
5. 用 PCA sketch 后再拟合 transfer；
6. 不改变 training rule。
```

如果 `NoiseSignalLeak` 过高：

```text
1. 检查 shuffled-label residual 构造；
2. 收紧 signal projector top-energy threshold；
3. 检查 CEp99/ECE 是否同步恶化；
4. 将该 candidate 标记为 GeometryBlocked，不做 CE tune。
```

如果 `RealSignalReservoirRatio` 高：

```text
1. 定位真实 residual 的 reservoir eigen components；
2. 检查早期 AUC-step 是否慢；
3. 将该信息传给 functional P3 reservoir-release candidate；
4. 不按 dataset 分支。
```

## 6.7 必须可视化

```text
fig_coupling_predicted_vs_actual.svg
fig_coupling_r2_by_candidate.svg
fig_signal_spectrum_by_candidate.svg
fig_real_signal_reservoir_ratio.svg
fig_noise_signal_leak.svg
fig_kernel_drift_vs_coupling.svg
fig_noise_leak_vs_ece.svg
fig_real_signal_reservoir_vs_auc_step.svg
```

---

# 7. Phase P3：Functional official re-entry, cloned one-step / five-step audit

## 7.1 目标

在 B320 locked / near-locked base 上，重新验证 functional update。

Functional 不能再是：

```text
generic output perturbation；
residual weight 小网格；
branch damping 小修；
未来动作 selector；
只看 train loss 的更新。
```

v12.10 的 functional update 必须是：

$$
\boxed{
\text{task-safe, signal-channel-aware, control-resistant geometry maintenance。}
}
$$

## 7.2 候选 functional mechanisms

### F1：AdamW-orthogonal signal residual

先取 AdamW / ManualAdamW 方向 $d_A$，functional 方向 $d_F$ 必须去掉与 $d_A$ 的平行部分：

$$
\tilde d_F=d_F-\frac{\langle d_F,d_A\rangle}{\|d_A\|^2+\epsilon}d_A.
$$

目标是测试 functional 是否能提供 AdamWParallel 无法提供的互补几何。

接受条件：

```text
cos(tilde_d_F, d_A) near 0
holdout descent non-harm
Line C improvement over base
beats AdamWParallelDirection
```

### F2：Reservoir-release update

当 Line C 显示真实信号困在 reservoir：

$$
RealSignalReservoirRatio\text{ high},
$$

functional update 目标是降低该比例，而不是单纯降 train loss。

接受条件：

$$
RealSignalReservoirRatio_{after}\le RealSignalReservoirRatio_{before}-0.02.
$$

同时：

$$
NoiseSignalLeak_{after}\le NoiseSignalLeak_{before}+0.005.
$$

### F3：Noise-leak vetoed geometry maintenance

任何 functional update 如果提高噪声进入 signal channel，直接拒绝：

$$
NoiseSignalLeak_{after}>NoiseSignalLeak_{before}+0.02
\Rightarrow reject.
$$

它不是 value source，而是 safety veto。

### F4：Tail-safe geometry maintenance

Functional update 必须不伤 hard tail：

$$
CEp99_{after}\le CEp99_{before}+\epsilon_{tail},
$$

$$
MarginP10_{after}\ge MarginP10_{before}-\epsilon_{margin}.
$$

## 7.3 Strong controls

每个 functional candidate 必须和以下 controls 同 batch / 同 checkpoint / 同 norm budget 比较：

```text
C0 TaskOnlyAdamW
C1 NoOpMatchedOverhead
C2 RandomMatchedNorm
C3 AdamWParallelDirection
C4 SNR-only
C5 GeometryOnlyNoSNR
C6 ShuffledPayload
C7 ShuffledEvent
C8 MLPAnalogGeometryMaintenance
C9 B320-base-no-functional-replay
```

## 7.4 必须记录

`v1210_functional_one_five_step.csv`：

```text
run_id
candidate_id
functional_id
control_id
dataset
seed
checkpoint_step
window_size
update_norm
update_norm_ratio_vs_adamw
cos_with_adamw
cos_after_orthogonalization
train_descent
probe_descent
holdout_descent_ratio
bad_step
bad_step_reason
CouplingR2_before
CouplingR2_after
CouplingR2_delta
RealSignalReservoirRatio_before
RealSignalReservoirRatio_after
RealSignalReservoirRatio_delta
NoiseSignalLeak_before
NoiseSignalLeak_after
NoiseSignalLeak_delta
CEp99_delta
ECE_delta
margin_p10_delta
step_overhead_ratio
control_gap
bootstrap_ci_low
official_candidate_pass
```

## 7.5 P3 official diagnostic gate

Functional candidate 进入 P4 short-run 必须满足：

$$
holdout\_descent\_ratio\ge 0.95,
$$

$$
bad\_step\_rate\le 0.02,
$$

$$
CouplingR^2_{func}\ge CouplingR^2_{base}+0.02,
$$

$$
NoiseSignalLeak_{func}\le NoiseSignalLeak_{base},
$$

$$
control\_gap\ge 0.005
$$

或：

$$
CI_{95,lower}(control\_gap)\ge0.
$$

如果只是 `delta_score_gap = 0.001` 这种弱正，不进入 official short-run，只标记为 diagnostic positive。

## 7.6 失败后的 Codex 自动尝试方向

### Case F-F1：Functional beats base but not AdamWParallel

说明方向仍被 AdamW 解释。Codex 先尝试：

```text
1. AdamW-orthogonal projection；
2. role-wise orthogonalization；
3. only apply to residual geometry subspace；
4. 不扫 weight 小网格。
```

### Case F-F2：Functional improves CouplingR2 but raises NoiseSignalLeak

说明把噪声推入 signal channel。Codex 先尝试：

```text
1. noise-leak veto；
2. signal projector threshold tightening；
3. shuffled residual negative control；
4. reject if no improvement。
```

### Case F-F3：Functional task-safe but no Line C improvement

说明只是无害扰动。Codex 先尝试：

```text
1. reservoir-release target；
2. tail-safe target；
3. basis/role-specific target；
4. 如果仍无 improvement，停止该 family。
```

### Case F-F4：Functional bad-step rate high

Codex 先尝试：

```text
1. backtracking lambda；
2. norm budget downscale；
3. event frequency downscale；
4. 如果仍 bad，不进入 short-run。
```

## 7.7 可视化

```text
fig_functional_control_gap.svg
fig_functional_delta_linec_vs_controls.svg
fig_functional_cos_with_adamw.svg
fig_functional_noise_leak_veto.svg
fig_functional_reservoir_release.svg
fig_functional_bad_step_rate.svg
fig_functional_predicted_vs_actual_descent.svg
```

---

# 8. Phase P4：Functional short-run official confirmation

## 8.1 开启条件

只有 P3 diagnostic gate 通过才运行 P4。不能因为 B320 base pass 就自动打开 functional short-run。

## 8.2 实验设计

```text
datasets = MNIST,Fashion-MNIST,KMNIST
seeds = 0,1,2 initially
train_size = 1024 or 2048
val_size = 512
test_size = 512
epochs = 3 or 5
batch_size = 128
```

方法：

```text
B320-AdamW
B320-NoOpMatchedOverhead
B320-RandomMatchedNorm
B320-AdamWParallelMaintenance
B320-SNR-only
B320-bestFunctional
MLP-AdamW
MLP-analogFunctional
```

## 8.3 必须记录

`v1210_functional_short_run.csv`：

```text
method
dataset
seed
acc_delta_vs_B320
val_loss_delta_vs_B320
AUC_step_delta_vs_B320
AUC_time_delta_vs_B320
ECE_delta_vs_B320
NLL_delta_vs_B320
CEp99_delta_vs_B320
margin_p10_delta_vs_B320
CouplingR2_delta_vs_B320
RealSignalReservoirRatio_delta_vs_B320
NoiseSignalLeak_delta_vs_B320
functional_event_count
accepted_event_count
rejected_event_count
amortized_overhead_ratio
control_gap_vs_best
strict_pass
```

## 8.4 P4 official success gate

Functional short-run pass：

$$
Acc_{func}\ge Acc_{B320}-0.003,
$$

$$
AUCtime_{func}\le AUCtime_{B320},
$$

$$
ECE_{func}\le ECE_{B320}+0.005,
$$

$$
CEp99_{func}\le CEp99_{B320}+\epsilon_{tail},
$$

$$
CouplingR^2_{func}\ge CouplingR^2_{B320}+0.02,
$$

$$
NoiseSignalLeak_{func}\le NoiseSignalLeak_{B320}-0.01,
$$

$$
amortized\_overhead\le1.05,
$$

并且：

$$
control\_gap\ge0.005
$$

或 bootstrap CI lower bound $\ge0$。

---

# 9. Phase P5：Classic family active branch, excluding B-spline

## 9.1 目标

经典 family 不是主线替代，而是 portfolio closure。每个 active family 必须输出状态与根因。B-spline 不参与。

`v1210_family_status.json` 必须输出：

```text
Rational.status
Chebyshev.status
Fourier.status
RBF_FastKAN.status
Wavelet.status
BSpline.status = FamilyFrozen_KernelBlocked_RejectedForThisVersion
```

## 9.2 Rational：TaskGeometry / CouplingCollapse repair

### 当前状态

Rational 已有若干 L3/A4 候选，但 A5 和 Line C 失败。已排除：

```text
pairNorm alone
hiddenRatResidual005 damping
batch-centered cap
trainable w1
shared low-rank pair signal
linear residual gain downscale
PCA whitening
PCA rotation/truncation
output-scale geometry partial repairs
```

核心 blocker：

```text
TaskBlocked / GeometryBlocked
coupling_collapse
AUC/ECE failure
not denominator unsafe
```

### v12.10 Rational hypotheses

#### R-H1：Rational 需要 stable task backbone，而不是 pure pair readout

尝试 Rational residual-on-B320 diagnostic：

```text
RAT-R1: B320 backbone + small Rational residual branch, frozen B320 for first stage
RAT-R2: B320-like direct/hinge branch + Rational group residual
RAT-R3: Rational branch only late-enabled after B320 reaches stable margin
```

目的：判断 Rational 是否可以作为 geometry/expression residual，而不是完整 base。

#### R-H2：Rational coupling collapse 来自 feature conditioning / readout geometry

尝试：

```text
RAT-C1: signal-sketch-conditioned readout initialization, unlabeled train-stream only
RAT-C2: readout spectral clipping, no CE tune
RAT-C3: group diversity regularized initialization, not loss penalty
RAT-C4: tangent-metric diagnostic for numerator/denominator, no direct functional claim
```

### Rational 必须记录

```text
L3 step_ratio_q90
memory_ratio_q90
grad_relerr_max
A4 expression pass
E1/E2/E6/E8 frozen R2
A5 mean/worst/near/ECE/AUC
CouplingR2
NoiseSignalLeak
RealSignalReservoirRatio
den_min / den_p01 / den_condition
r_prime_p95 / r_double_prime_p95
group_function_diversity
group_dead_fraction
basis_condition_proxy
```

### Rational pass gate

$$
step\_ratio_{q90}\le1.10,
$$

$$
memory\_ratio_{q90}\le0.80,
$$

$$
A4=1,
$$

$$
mean\_delta\ge0,
$$

$$
worst\_delta\ge-0.010,
$$

$$
near\_rate\ge0.90,
$$

$$
AUCstep\le1.00,
$$

$$
CouplingR^2\ge CouplingR^2_{MLP}-0.02.
$$

If L3+A4 pass but A5/Line C fail：`TaskBlocked` or `GeometryBlocked`。
If L3 fail：`KernelBlocked`。
If A4 fail：`ExpressionBlocked`。

## 9.3 Chebyshev：L3 exists, expression repair without cost explosion

### 当前状态

Chebyshev 有 L3 fused measured path，部分候选 step 约 $1.12$-$1.27$，但 A4 expression fail。

### v12.10 Chebyshev experiments

```text
CHEB-E1: K3 + linear direct skip + fused readout
CHEB-E2: K4 low-degree expansion, no high-degree free growth
CHEB-E3: paircrossR16/R32 expression repair with fused gradbuf
CHEB-E4: degree-energy damping initialization, no loss penalty
CHEB-E5: B320-like cheap direct branch + Chebyshev residual diagnostic
```

### Chebyshev metrics

```text
L3 step/memory/backward/update
A4 expression battery
Cheb degree_energy_k
high_degree_energy_ratio
recurrence_max_abs
CouplingR2
NoiseSignalLeak
CEp99/ECE
```

### Chebyshev pass gate

$$
step\_ratio_{q90}\le1.20,
$$

$$
memory\_ratio_{q90}\le1.00,
$$

$$
A4=1,
$$

then A5 triage allowed.

If expression still fails after K4 + paircrossR32 while step remains pass，mark `ExpressionBlocked` and stop wider grids.

## 9.4 Fourier：low-frequency expression repair

### 当前状态

Fourier low-frequency fused L3 can be close to efficiency gate, but A4 expression fails.

### v12.10 Fourier experiments

```text
FOUR-E1: K2 fixed-frequency + linear direct branch
FOUR-E2: K4 fixed-frequency + fused sincos
FOUR-E3: lowfreq + cheap quadratic residual
FOUR-E4: late-enable high-frequency diagnostic, not official base
FOUR-E5: frequency-energy damping initialization
```

No learnable frequency / phase in first stage.

### Fourier metrics

```text
sincos_count
spectral_entropy
high_freq_energy_ratio
phase_drift, if any
A4 expression
Line C NoiseSignalLeak
CEp99/ECE
```

### Fourier pass gate

$$
step\_ratio_{q90}\le1.20,
$$

$$
A4=1,
$$

$$
NoiseSignalLeak\le NoiseSignalLeak_{MLP}+0.02.
$$

If A4 improves but NoiseSignalLeak high，mark `GeometryBlocked` rather than CE tune.

## 9.5 RBF / FastKAN：compact local only

### 当前状态

RBF current representative is KernelBlocked. Dense RBF is not allowed.

### v12.10 RBF experiments

```text
RBF-F1: fixed uniform centers, fixed width, K_active=2
RBF-F2: fixed uniform centers, fixed width, K_active=4
RBF-F3: FastKAN-style Gaussian approximate spline, no dense basis materialization
RBF-F4: exp2 / polynomial exp approximation diagnostic
```

禁止：

```text
dense [B,d,K] materialization
learnable center/width in first stage
dense all-center evaluation
```

### RBF metrics

```text
K_total
K_active
exp_count_per_sample
active_center_entropy
dead_center_fraction
out_of_grid_fraction
width_p01 / width_p99, if width learned later
L3 step/memory
A4/A5/Line C
```

### RBF pass gate

$$
step\_ratio_{q90}\le1.25,
$$

$$
memory\_ratio_{q90}\le1.00,
$$

then A4 allowed.

If K_active=2/4 still fails L3，mark `KernelBlocked` and stop.

## 9.6 Wavelet：low-priority local hat/triangle only

### 当前状态

Wavelet current representative is severely KernelBlocked. v12.10 only keeps the cheapest local wavelet variants.

### v12.10 Wavelet experiments

```text
WAV-F1: hat / triangle local wavelet, reuse RBF local kernel infrastructure
WAV-F2: Haar/step diagnostic, no smooth claim
WAV-F3: Morlet/MexicanHat disabled unless local hat passes L3
```

### Wavelet pass gate

$$
step\_ratio_{q90}\le1.25,
$$

$$
memory\_ratio_{q90}\le1.00.
$$

If hat/triangle fails，mark `KernelBlocked` and freeze Wavelet for this version.

## 9.7 B-spline：frozen

No candidate scheduling. No kernel implementation. No Codex search.

`v1210_family_failure_table.csv` must contain：

```text
family = BSpline
status = FamilyFrozen_KernelBlocked_RejectedForThisVersion
reason = repeated KernelBlocked; L3 manual step 1.5046; memory 1.4452; no task/geometry evidence; engineering cost not justified in v12.10
reactivation_condition = external fused forward+backward/update kernel passes L3
```

---

# 10. Phase P6：Parallel execution schedule

v12.10 必须并行，不能串行等所有 family 完成。

## Batch 0：P0 contract

```text
Run P0 route/family/provenance contract。
Block all downstream if B-spline active candidate appears。
```

## Batch 1：B320 base and Line C in parallel

```text
Job A1:
  B320/B321/B314 exact 10-seed hardening。

Job C1:
  B320/B321 exact Line C early/mid/late windows。

Job D1:
  Rational B7 best-current autopsy replay with Line C。

Job D2:
  Chebyshev/Fourier minimal expression repair smoke。
```

## Batch 2：Functional cloned audit

Start only after B320 exact pass or strong near-pass.

```text
Job B1:
  F1/F2/F3/F4 functional one-step/five-step.

Job C2:
  Line C for functional + controls.
```

## Batch 3：Classic family triage

```text
Rational structural mechanisms。
Chebyshev/Fourier expression repair。
RBF compact local L3。
Wavelet hat local L3.
```

B-spline not scheduled.

## Batch 4：Short-run confirmation

Only for candidates that pass diagnostic gates:

```text
B320 final base confirm
Functional short-run if P3 pass
Classic family A5 if L3+A4+LineC eligible
```

---

# 11. Unified artifact contract

v12.10 must output:

```text
v1210_route_decision.json
v1210_family_policy.json
v1210_provenance_audit.csv
v1210_b320_base_hardening.csv
v1210_b320_task_trace.csv
v1210_b320_efficiency_profile.csv
v1210_linec_geometry.csv
v1210_functional_one_five_step.csv
v1210_functional_short_run.csv
v1210_family_manifest.csv
v1210_family_efficiency.csv
v1210_family_gradcheck.csv
v1210_family_expression.csv
v1210_family_task_triage.csv
v1210_family_linec.csv
v1210_family_failure_table.csv
v1210_family_status.json
v1210_hash_manifest.json
```

`v1210_route_decision.json` must include:

```text
base_anchor_candidate
base_hardening_pass
linec_nontearing_pass
functional_official_success
classic_family_pass_count
bspline_frozen
no_fake_provenance_pass
next_recommended_action
```

---

# 12. Required visualizations

## Base / efficiency

```text
fig_b320_efficiency_stack.svg
fig_b320_task_delta_heatmap.svg
fig_b320_auc_step_time_heatmap.svg
fig_b320_protocol_stress_pareto.svg
fig_b320_vs_b314_b321_radar.svg
```

## Line C

```text
fig_coupling_r2_by_candidate.svg
fig_signal_spectrum_by_candidate.svg
fig_noise_signal_leak_by_candidate.svg
fig_real_signal_reservoir_ratio_by_candidate.svg
fig_kernel_drift_vs_coupling.svg
fig_linec_vs_auc_step.svg
```

## Functional

```text
fig_functional_control_gap.svg
fig_functional_delta_score_bootstrap.svg
fig_functional_noise_leak_veto.svg
fig_functional_reservoir_release.svg
fig_functional_task_nonharm.svg
fig_functional_overhead.svg
```

## Classic family, excluding B-spline

```text
fig_classic_family_status_no_bspline.svg
fig_classic_efficiency_pareto_no_bspline.svg
fig_classic_expression_radar_no_bspline.svg
fig_rational_coupling_collapse_autopsy.svg
fig_cheby_fourier_expression_vs_efficiency.svg
fig_rbf_wavelet_kernel_blocker.svg
```

B-spline should appear only as a frozen row in the family status figure, not as an active plotted candidate.

---

# 13. Final decision rules

## Case A：B320 hardening pass, functional fail

结论：

```text
B320 becomes official strict efficient PureKAN base anchor。
Functional remains unsolved。
Next plan focuses on functional mechanism, not base search。
```

## Case B：B320 hardening pass, functional P3 pass but P4 fail

结论：

```text
Functional has diagnostic mechanism but not training success。
Need short-run failure autopsy: task nonharm, control gap, Line C drift。
```

## Case C：B320 hardening + functional P4 pass

结论：

```text
First official DG-KAN success candidate。
Proceed to 10-seed functional confirmation and external fair protocol。
```

## Case D：B320 fails hardening

结论：

```text
Do not open functional official。
Run base failure autopsy: AUC-step vs AUC-time vs Line C vs efficiency。
Do not return to B-spline。
```

## Case E：Rational or Cheby/Fourier family pass

结论：

```text
Add as alternative base candidate。
Compare against B320 on task, efficiency, Line C, functional compatibility。
Do not replace B320 unless it beats B320 under same gates。
```

## Case F：all classic active families fail

结论：

```text
Classic portfolio remains unresolved or rejected for this version。
B-spline remains frozen。
Continue B320 + functional mainline。
```

---

# 14. v12.10 的最终定位

v12.10 的核心不是“继续找 base”，也不是“继续平均扫所有 basis”。它的定位是：

$$
\boxed{
\text{以 B320 为当前最强 base anchor，}
\text{用 Line C 证明几何健康，}
\text{用 strong controls 证明 functional update 的独立价值。}
}
$$

经典 basis 支线仍保留，但排除 B-spline：

```text
B-spline:
  frozen, no active work.

Rational:
  priority classic family, because it已经进入 L3/A4 后的 Task/Geometry blocker。

Chebyshev/Fourier:
  priority expression repair families, because已有或接近 L3 fused path。

RBF/FastKAN:
  compact-local-only efficiency test。

Wavelet:
  low-priority local-hat-only diagnostic。
```

一句话总结：

$$
\boxed{
\text{B320 负责把 base 闭合；functional update 负责证明核心研究贡献；}
\text{经典 basis 支线保留但不拖慢主线；B-spline 在 v12.10 冻结。}
}
$$
