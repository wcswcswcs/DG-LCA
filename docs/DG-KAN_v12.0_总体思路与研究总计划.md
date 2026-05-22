# DG-KAN v12.0 总体思路与研究总计划：Repaired-LQ Good Geometry Battery 与 Task-Safe Functional Geometry Maintenance

版本：v12.0  
目标读者：项目研究者、实验执行者、Codex 实现者  
核心要求：Typora 友好公式；全文公式只使用 `$...$` 与 `$$...$$`。

---

## 0. 一句话主线

DG-KAN 当前不应该继续被定义为“找一个更聪明的 functional update 小技巧”。真正的主线应该重新锚定为：

$$
\boxed{
\text{合格的 strict FC-PureKAN base}
+
\text{task-safe functional geometry maintenance}
\rightarrow
\text{比 PureKAN+ordinary backprop 更好的模型}
}
$$

这里的“更好”必须有硬约束，不允许用单一 accuracy、单一 smoothness、单一抗遗忘或单一小数据集胜利来替代。我们要做的是 next-generation MLP alternative，因此必须同时满足：

1. **表达力不打折**：PureKAN base 至少达到 MLP envelope，最好在 interaction / composition / low-order nonlinear target 上略强。
2. **系统效率不崩**：forward、backward、step time 与 memory 必须在 MLP-comparable envelope 内。
3. **收敛不慢**：按 step 与 wall-clock 同时看，不能只看最终 accuracy。
4. **几何更健康**：训练轨迹更少撕裂数据流形附近的函数，cover / rank / basis usage 不塌，tail 不恶化。
5. **functional update 有独立贡献**：必须在同一个 PureKAN base 上击败 task-only、NoOp、RandomMatchedNorm、AdamWParallelDirection、SNR-only、MLP analog、QuadraticFeatureMLP analog 等 controls。
6. **抗噪声和抗遗忘是加分项，不是第一主线**：不能为了抗遗忘牺牲表达、速度、收敛和公平比较。

因此当前主线不是：

```text
继续找 action；
继续训练 future selector；
继续做 dataset-specific controller；
继续扩大 v10 future-path / action materializer；
继续把 functional update 当成主 optimizer；
继续用 low curvature 当几何成功。
```

当前主线是：

```text
repaired LQ / LQ-t2 base
-> Good Geometry Battery
-> GeometryCertificateV0
-> task-safe functional geometry maintenance
-> control-resistant paired proof
-> 10-seed / external fair confirmation
```

---

## 1. 当前状态的事实判断

### 1.1 已经站住的地基：LQ / repaired LQ 是当前唯一主 base

旧 DWM2、Dense RBF、AB-RBF、SparseInterp、RationalKAT、U-FULL 等路线都提供了机制经验，但当前真正应该作为主 base 的是 repaired LQ / LQ-t2 系列，尤其是 `R2-LQ-fanin-output-scale-confirmed`。

当前 LQ 主线的意义不是“已经 Beyond MLP”，而是：

```text
第一次出现了 strict FC-PureKAN 在系统效率和任务 trainability 上接近 MLP 的 base。
```

其中历史 LQ-t2 / `LinearLiftQuadraticEdgeBasis` 已经证明了几个关键点：

```text
strict FC-PureKAN path 成立；
Linear lift + quadratic edge basis 能保留 pairwise interaction；
compact / recompute live-set 下 P4 system gate 成立；
三任务三 seed 达到 robust near-pass，但不是 full pass；
conservative memory 仍需报告，不能隐藏。
```

后续 repaired LQ 又把 base 稳定性推进了一步：

```text
R2-LQ-fanin-output-scale-confirmed 达到 9/9 near rows；
macro delta 约 -0.003；
step q90 约 1.01；
不依赖 dataset-specific route。
```

因此本计划把主 base 固定为：

```text
Primary base:
  R2-LQ-fanin-output-scale-confirmed / repaired LQ

Historical/current reference:
  LQ-t2-h256 / LinearLiftQuadraticEdgeBasis

Diagnostic baselines:
  same-param MLP
  same-FLOPs / same-step MLP
  QuadraticFeatureMLP
  D2 compositional FullEdge interaction diagnostic
```

D2 compositional FullEdge 不能重新成为主线。它证明过 interaction 和 composition 很重要，但历史上 P4 system gate 没闭合；它应该作为 expression / interaction diagnostic，而不是压过 LQ / repaired LQ。

### 1.2 functional update 还没有成功

当前最容易犯的错误是：因为我们已经有 functional actuator、carrier、snapshot attach、value score 的局部推进，就写成 functional update 成功。这是错误的。

当前已知事实更准确地说是：

```text
1. P4-qualified actuator 已经不是 blocker。
2. carrier / attach / no-event preservation 已经有明显推进。
3. 但是 RealFunctional 在 paired replay 中没有聚合击败 matched controls。
4. v9.2.40 之后真正失败点集中到 value observability / control-gap mismatch。
5. v9.2.41 局部 legal score 有进步，但 leave-dataset-out / leave-stratum-out / official paired replay 仍未形成 strict functional success。
```

所以后续不能继续写：

```text
functional update 已经证明有独特优势；
carrier 能动就等于 functional 成功；
局部 value score 过了就能开 full-run；
KMNIST 某些 row 有 beat 就说明 functional 成功。
```

正确写法是：

$$
\boxed{
\text{functional actuatability 已部分解决；当前 blocker 是可验证、可泛化、control-resistant 的 value / geometry certificate。}
}
$$

### 1.3 Deep Manifold 与 Generalization paper 的作用

这些论文给了我们重新定义“几何好”的语言和计算灵感，但不能直接当成 DG-KAN 已经成功的证明。

Deep Manifold 的有用启发是：神经网络训练不应被看成固定光滑流形上的普通参数优化，而更像不断移动的 local covers / stacked piecewise manifolds / residual-driven fixed-point regions。它提醒我们，几何好不等于 coefficient curve 平滑，而是：

```text
cover 不塌；
local patch 之间转换稳定；
扰动后函数漂移更小；
训练轨迹能形成稳定 fixed-point-like regions；
局部曲率不过度积累。
```

Generalization theory 的有用启发是：输出空间可以分成 signal channel 和 reservoir。不是所有训练集 noise fitting 都同样有害；坏的是 noise 被推入 test-visible signal channel。它给了我们一个更可执行的判据：microbatch gradient 的均值平方必须强过方差项。

一个简化的一维 per-parameter / per-role 形式是：

$$
\mu_k^2 > \frac{\sigma_k^2}{b-1}.
$$

更一般地，对一个 metric 或 preconditioner $M$，one-step population-risk proxy 可以写成：

$$
\Omega_B(M)
=
\frac{1}{b(b-1)}
\sum_{a\ne c} r_a^T K^M_{ac} r_c.
$$

在参数空间的近似形式是：

$$
\Omega_B(M) \approx \operatorname{tr}\left(M A_B\right),
$$

其中：

$$
A_B = \bar g_B \bar g_B^T - \frac{1}{b-1}\Sigma_B.
$$

这给我们的启发是：functional update 不应直接被设计成另一个 task-descent optimizer，而应被设计成：

$$
\boxed{
\text{只在 population-safe signal channel 中运行的 geometry maintenance。}
}
$$

---

## 2. 项目总目标

### 2.1 Minimum claim

最低可发表或可继续扩展的 claim 是：

$$
\boxed{
\text{在 strict FC-PureKAN repaired LQ base 上，functional geometry maintenance 可以在不损失表达、任务、速度的前提下，稳定改善 geometry / calibration / tail-risk 中至少一项，并击败强 controls。}
}
$$

这要求：

```text
LQ+functional >= LQ-AdamW - tolerance；
ValLossAUC_time 不差；
ECE / NLL 不恶化；
GeoCertificate 至少一个核心维度改善；
NoOp / Random / AdamWParallel / SNR-only / Geometry-only / Shuffled controls 不能解释；
MLP analog 与 QuadraticFeatureMLP analog 不能完全解释。
```

### 2.2 Strong claim

强 claim 是：

$$
\boxed{
\text{LQ+functional 在 task、ValLossAUC_time、ECE/NLL、GeometryCertificate 上系统性优于 LQ+AdamW。}
}
$$

这要求 paired multi-seed 结果成立，而不是单 seed 或单数据集局部成立。

### 2.3 External-ready claim

最终 external-ready claim 是：

$$
\boxed{
\text{在公平参数量、FLOPs、wall-clock、memory、CE-only、no teacher、no loss modification 条件下，LQ+functional 达到或超过 MLP task envelope，并在 geometry / calibration / robustness 上有明确优势。}
}
$$

这时才能开始认真讨论 next-generation MLP alternative。

---

## 3. 核心假设

本计划围绕假设设计实验，不围绕“哪个 runner 看起来赢”设计实验。

### H1：repaired LQ 是当前可用的 strict FC-PureKAN base

假设：`R2-LQ-fanin-output-scale-confirmed` 能在 global protocol 下稳定保持 near-pass、P4 system envelope、strict PureKAN contract，不需要 dataset-specific 修补。

成立标准：

$$
near\_pass\_rate \ge 0.80,
$$

$$
\Delta Acc_{macro} \ge -0.005 \text{ 或至少 } \ge -0.01,
$$

$$
step\_q90 \le 1.10,
$$

$$
memory_{compact} \le 1.00,
$$

并且 conservative memory 必须报告，不允许删除失败 accounting。

如果 H1 不成立，functional update 全部暂停，回到 base repair。

### H2：KAN base 本身可能提供 geometry advantage

假设：即使不用 functional update，LQ / repaired LQ 也可能因为 edge-basis interaction 和 lift-coordinate 结构，在某些 geometry metrics 上优于 same-param MLP。

成立标准：比较：

```text
MLP-AdamW
QuadraticFeatureMLP-AdamW
LQ-AdamW / repaired LQ-AdamW
```

若 LQ 在 expression battery、cover/rank、perturbation drift、tail stability、ECE/NLL、signal consistency 中至少部分优于 MLP，并且 task / efficiency 不输，则可称为 KAN base contribution。

### H3：好几何是 Pareto certificate，不是单一 smoothness score

假设：几何好必须被定义为一组 hard gates + Pareto 改善，而不是手工加权分数。

我们定义：若 $
\theta'
$ 的几何优于 $
\theta
$，记作：

$$
\theta' \succ_G \theta.
$$

它至少要求：

$$
\operatorname{Expressivity}(\theta') \ge \operatorname{Expressivity}(\theta) - \epsilon_E,
$$

$$
\operatorname{TrainabilityCondition}(\theta') > \operatorname{TrainabilityCondition}(\theta),
$$

$$
\operatorname{SignalNoiseSeparation}(\theta') > \operatorname{SignalNoiseSeparation}(\theta),
$$

$$
\operatorname{NoHarm}(\theta',\theta)=1,
$$

$$
\operatorname{Cost}(\theta'\leftarrow\theta) \le C_{max}.
$$

如果曲率下降但表达力塌缩，不算好几何。若 task loss 短期下降但 tail、ECE、rank、basis coverage 恶化，也不算好几何。

### H4：functional update 应该是低频、task-safe、signal-projected geometry maintenance

假设：functional update 不应该替代 AdamW / ManualAdamW 的 task descent，而应作为小幅 correction：

$$
\theta_{t+1}
=
\theta_t
+
\Delta\theta_{task}
+
\lambda_t \Delta\theta_{geo}.
$$

其中：

```text
Delta theta task:
  AdamW / ManualAdamW / stable task optimizer。

Delta theta geo:
  cover / curvature / perturbation / calibration / tail geometry maintenance。

lambda_t:
  train-stream probe + backtracking + safety gate 决定。
```

Functional correction 成立标准：

$$
\cos(d_{new}, d_{task}) \ge 0.85,
$$

$$
\frac{\operatorname{HoldoutDescent}(d_{new})}{\operatorname{HoldoutDescent}(d_{task})} \ge 0.95,
$$

$$
\operatorname{BadStepRate}(d_{new}) \le 0.02,
$$

并且至少一个 geometry / calibration / tail metric 改善。

### H5：SNR / offdiag 是 safety / projection，不是价值源

假设：SNR / offdiag 可以阻止 noise 进入 signal channel，但不能单独提供 functional 贡献。它应该作为：

```text
veto gate；
projection metric；
risk penalty；
accepted-direction diagnostic。
```

而不是写成：

```text
SNR 高 -> action 好。
```

### H6：优势必须被 factorial design 拆开

需要同时拆分：

```text
KAN base contribution；
functional contribution；
KAN-specific functional synergy；
random maintenance control；
MLP analog control；
QuadraticFeatureMLP control。
```

设：

```text
A: MLP + AdamW
B: LQ / repaired LQ + AdamW
C: LQ / repaired LQ + AdamW + functional geometry maintenance
D: MLP + AdamW + analogous geometry maintenance
E: LQ + AdamW + same-drift random maintenance
F: LQ + AdamW + AdamWParallelDirection maintenance
G: QuadraticFeatureMLP + analogous geometry maintenance
```

则：

$$
\text{KAN base contribution} = B-A,
$$

$$
\text{functional-on-KAN contribution} = C-B,
$$

$$
\text{KAN-specific functional synergy} = (C-B)-(D-A),
$$

$$
\text{random-control adjusted contribution} = C-E.
$$

只有 $C$ 击败 $B,E,F,D,G$，才能说 functional update 有独立价值。

---

## 4. Good Geometry Battery 的正式定义

Good Geometry Battery 不用于直接训练，它先作为被动测量系统，之后才作为 functional maintenance 的 candidate objective。它分为六类。

### 4.1 Expressivity and cover capacity

#### Effective rank

对 hidden / lift / output representation $Z$ 做 SVD，奇异值为 $\sigma_i$：

$$
p_i=\frac{\sigma_i}{\sum_j \sigma_j + \epsilon}.
$$

有效秩：

$$
R_{eff}(Z)
=
\exp\left(-\sum_i p_i\log(p_i+\epsilon)\right).
$$

要求：

$$
R_{eff}(Z_{\theta'}) \ge R_{eff}(Z_\theta)-\epsilon_R.
$$

#### Expression battery

必须测：

```text
additive target；
pairwise product target；
composition target；
local XOR target；
high-frequency target；
noise-stress target。
```

目的不是在这些 toy target 上打榜，而是判断 LQ / PureKAN 是否真的保留 MLP 不易直接表达的 interaction / composition 能力。

#### Basis / cover usage

对 LQ 的 lift coordinate 和 quadratic basis 记录：

```text
lift_effective_rank；
lift_condition_proxy；
basis_usage_entropy；
dead_basis_fraction；
channel_norm_ratio；
quadratic_coeff_norm；
output_linear_norm；
role_update_share。
```

basis occupancy entropy：

$$
H_{occ}=-\sum_k p_k\log(p_k+\epsilon).
$$

若 $H_{occ}$ 降低、dead basis 增加、effective rank 下降，即使 curvature 下降，也不能称为好几何。

### 4.2 Trainability condition

对 reference set $R$，输出对参数的 Jacobian 为 $J_R$，输出 kernel 为：

$$
K_R=J_RJ_R^T.
$$

稳健条件数 proxy：

$$
\kappa_{robust}(K_R)=\frac{\lambda_{90}(K_R)+\epsilon}{\lambda_{10}(K_R)+\epsilon}.
$$

Top eigen share：

$$
TopShare(K_R)=\frac{\lambda_1}{\sum_i\lambda_i+\epsilon}.
$$

目标：

```text
robust condition 下降；
top eigen share 不过度集中；
rank 不塌；
update-to-output ratio 稳定；
gradient spike 降低。
```

### 4.3 Local manifold / perturbation stability

对输入扰动 $\epsilon v$，定义局部 sensitivity：

$$
S_{local}(x,v)=\frac{\|f_\theta(x+\epsilon v)-f_\theta(x)\|_2}{\epsilon\|v\|_2+\epsilon_0}.
$$

扰动收缩比：

$$
C_{perturb}
=
\frac{\operatorname{median}_{x,v} S_{local}^{after}(x,v)}
{\operatorname{median}_{x,v} S_{local}^{before}(x,v)+\epsilon}.
$$

augmentation logit drift：

$$
D_{aug}^{p95}=Q_{0.95}\left(\|f_\theta(a(x))-f_\theta(x)\|_2\right).
$$

目标：

```text
perturbation contraction ratio 下降；
augmentation logit drift 下降；
local Jacobian p95 不爆；
data-tangent curvature 不恶化。
```

### 4.4 Curvature debt

对 edge / basis coefficient，用二阶差分近似：

$$
D^2 c_k=c_{k+1}-2c_k+c_{k-1}.
$$

Curvature debt：

$$
CurvatureDebt
=
\sum_{e,k}p_{e,k}\left(c_{e,k+1}-2c_{e,k}+c_{e,k-1}\right)^2.
$$

对 LQ 这类低阶 basis，也要记录 quadratic channel 与 lift channel 的 curvature-like proxy，不要把它作为唯一几何结论。

### 4.5 Signal / noise separation

把 batch 切成 $S$ 个 microbatches，每个得到输出更新或梯度方向 $u_s$：

$$
\bar u=\frac{1}{S}\sum_s u_s.
$$

Signal consistency：

$$
SignalConsistency=
\frac{\|\bar u\|^2}
{\frac{1}{S}\sum_s\|u_s-\bar u\|^2+\epsilon}.
$$

Noise leak：

$$
NoiseLeak=\frac{Improve_{noise}}{Improve_{real}+\epsilon}.
$$

SNR gate：

$$
SNR_k=\mu_k^2-\frac{\sigma_k^2}{b-1}.
$$

记录：

```text
SNR_positive_fraction；
rolewise_SNR_positive_fraction；
offdiag_population_proxy；
signal_channel_projection_ratio；
noise_label_signal_projection；
reservoir_like_ratio。
```

### 4.6 Tail / calibration / no-harm

必须记录：

```text
CEp95 / CEp99；
margin_p10；
wrong_confidence_p95；
ECE；
NLL；
Brier；
hard-tail classwise error；
memory set loss drift；
logit drift p95。
```

函数漂移：

$$
D_{logit}^{p95}=Q_{0.95}\left(\|f_{\theta'}(x)-f_\theta(x)\|_2\right).
$$

memory loss drift：

$$
\Delta L_{memory}=L(\theta';M)-L(\theta;M).
$$

hard-tail drift：

$$
\Delta CE_{hard,p99}=CE_{p99}(\theta';H)-CE_{p99}(\theta;H).
$$

任何 functional maintenance 都必须满足：

```text
CEp99 不升；
margin_p10 不降；
ECE / NLL 不恶化；
memory loss drift 不恶化；
rank / basis coverage 不塌。
```

---

## 5. GeometryCertificateV0

GeometryCertificateV0 不直接给一个神秘总分。它由 hard gates 和 Pareto frontier 组成。

### 5.1 Hard gates

对 candidate $C$ 与 reference $B$：

$$
Acc_C \ge Acc_B - \tau_{acc},
$$

其中第一阶段：

$$
\tau_{acc}=0.005,
$$

exploratory 可放宽到：

$$
\tau_{acc}=0.01.
$$

Val loss AUC by time：

$$
ValLossAUC_{time,C}\le ValLossAUC_{time,B}+\tau_{auc}.
$$

Calibration：

$$
ECE_C\le ECE_B+\tau_{ece},
$$

$$
NLL_C\le NLL_B+\tau_{nll}.
$$

Efficiency：

$$
\frac{T_{step,C}}{T_{step,B}}\le 1.10
$$

或低频 event amortized：

$$
\frac{T_{amortized,C}}{T_{B}}\le 1.05.
$$

Rank / cover：

$$
R_{eff,C}\ge R_{eff,B}-\epsilon_R,
$$

$$
H_{occ,C}\ge H_{occ,B}-\epsilon_H.
$$

Tail：

$$
CEp99_C\le CEp99_B+\tau_{tail},
$$

$$
MarginP10_C\ge MarginP10_B-\tau_{margin}.
$$

### 5.2 Pareto geometry dimensions

一个 candidate 至少要在下面维度中形成非平凡改善：

```text
CurvatureDebt 下降；
PerturbDrift 下降；
KernelCondition 改善；
TopEigenShare 不过度集中；
SignalConsistency 上升；
NoiseLeak 下降；
CoverEntropy 不下降；
HardTailRisk 下降；
ECE / NLL 改善。
```

如果所有 geometry metric 都没有改善，即使 task 不坏，也不能称为 functional geometry success。

---

## 6. 后续总实验路线

### Phase 0：路线冻结与 no-fake contract

冻结：

```text
v10 future/action/controller 主线；
FastGood / SlowBurnGood selector；
GoodSubspace / GoodCone 作为几何定义；
function-space task descent oracle 作为主线；
旧 U-FULL full optimizer；
Dense RBF / AB-RBF efficiency claim；
dataset-specific threshold / route；
Conv / Former 扩展。
```

保留：

```text
repaired LQ base；
LQ-t2 historical reference；
MLP controls；
QuadraticFeatureMLP control；
Good Geometry Battery；
SNR/offdiag safety；
train-stream microprobe；
paired controls。
```

所有实验继续遵守：

```text
CE-only；
no teacher；
no distillation；
no loss modification；
no sampler / class weight；
no dataset-name branch；
no fake / proxy rows；
KAN path 不使用 PyTorch loss.backward graph；
functional update 是 update rule，不是 loss。
```

### Phase 1：Base qualification

目标：确认 repaired LQ 是当前 official base。

方法：

```text
A0 MLP same-param + AdamW
A1 MLP same-FLOPs / same-step + AdamW
A2 QuadraticFeatureMLP + AdamW
A3 LQ-t2-h256 + AdamW
A4 R2-LQ-fanin-output-scale-confirmed + AdamW
A5 D2 compositional FullEdge diagnostic
```

必须记录：

```text
strict PureKAN equivalence；
non-edge params；
manual forward/backward/update；
pairwise-product R2；
composition target R2；
local XOR target；
high-frequency target；
accuracy；
ValLossAUC_step/time；
ECE / NLL；
forward/backward/step ratio；
compact memory；
conservative memory。
```

通过标准：

```text
repaired LQ robust near-pass；
step ratio <= 1.10；
compact memory pass；
conservative memory reported；
expression battery 不输 MLP / QuadraticFeatureMLP；
不能靠 functional 才让 base 成立。
```

### Phase 2：Passive Good Geometry Battery

不改训练，只测：

```text
MLP-AdamW；
QuadraticFeatureMLP-AdamW；
LQ / repaired LQ-AdamW。
```

目标：判断哪些 geometry metrics 能预测：

```text
收敛快；
near-pass / full-pass；
ECE / NLL；
hard-tail；
KMNIST miss rows；
noise resistance。
```

Phase 2 结束后产出：

```text
geometry_snapshot.csv
geometry_checkpoint_trace.csv
geometry_correlation_report.csv
GeometryCertificateV0.json
```

### Phase 3：Functional maintenance one-step / five-step audit

只在 cloned weights 上做，不污染训练。

候选：

```text
C0 task-only AdamW / ManualAdamW
C1 NoOp matched overhead
C2 Random matched norm
C3 AdamWParallelDirection
C4 Shuffled payload / shuffled event
C5 SNR-only safety gate
C6 Geometry-only no SNR
C7 SNRProjectedGeometry
C8 BasisEntropyRebalance
C9 LiftConditionRepair
C10 TailStabilityCorrection
C11 MLP analog geometry maintenance
C12 QuadraticFeatureMLP analog maintenance
```

记录：

```text
train descent；
holdout descent；
holdout_descent_ratio；
bad_step_rate；
cos(d_new,d_task)；
functional norm ratio；
GeoCertificate delta；
ECE/NLL proxy delta；
CEp99 / margin-p10 delta；
rank / basis / cover delta；
noise-label delta；
amortized cost；
control beat rate。
```

通过条件：

$$
\frac{HoldoutDescent_{new}}{HoldoutDescent_{task}}\ge 0.95,
$$

$$
BadStepRate_{new}\le 0.02,
$$

$$
\cos(d_{new},d_{task})\ge 0.85,
$$

并且：

```text
GeoCertificate 至少一个核心维度改善；
beats NoOp / Random / AdamWParallel / Shuffled；
不能被 MLP analog / QuadraticFeatureMLP analog 完全解释。
```

### Phase 4：Short-run controlled training

只有 Phase 3 通过才跑 short training。

方法：

```text
LQ-AdamW
LQ-AdamW + NoOp overhead
LQ-AdamW + Random maintenance
LQ-AdamW + AdamWParallel maintenance
LQ-AdamW + SNR-only
LQ-AdamW + best functional geometry maintenance
MLP-AdamW
MLP + analogous geometry maintenance
QuadraticFeatureMLP + analogous maintenance
```

设置：

```text
datasets = MNIST, Fashion-MNIST, KMNIST
seeds = 0,1,2
budget = 20 或 30 epochs equivalent
```

通过条件：

```text
functional 不输 LQ-AdamW task；
ValLossAUC_time 不差；
ECE/NLL 或 hard-tail 改善；
GeometryCertificateV0 通过；
controls 不能解释；
step/memory envelope 不破。
```

### Phase 5：Stage-wise functional training

只有 short-run 通过才进入阶段化训练：

```text
Stage I:
  base task learning；functional off 或低频。

Stage II:
  plateau / geometry debt event 触发 maintenance。

Stage III:
  late consolidation；小步 geometry + calibration maintenance。
```

触发条件只能用 dataset-agnostic signals：

```text
geometry debt；
basis underuse；
lift condition；
hard-tail drift；
ECE/NLL drift；
SNR/offdiag safety；
train-stream holdout probe。
```

禁止：

```text
if dataset == KMNIST then use rule X；
为 Fashion / KMNIST / MNIST 单独调 threshold；
因为某个数据集过了就选择 score；
MNIST abstain rule。
```

### Phase 6：10-seed / external fair

只有 Phase 4/5 过，才跑：

```text
seeds = 0..9；
same-param MLP；
same-FLOPs MLP；
same-wall-clock MLP；
QuadraticFeatureMLP；
CE-only；
no teacher；
no sampler；
no loss modification。
```

最终报告拆成三张表：

```text
Base table:
  LQ vs MLP，证明 KAN base contribution。

Functional table:
  LQ+functional vs LQ，证明 functional contribution。

Synergy/control table:
  LQ+functional vs MLP analog / Random / AdamWParallel / QuadraticFeatureMLP，证明不是普通 geometry trick。
```

---

## 7. 可视化总要求

必须生成以下图，不允许只给 CSV。

### 7.1 Base and task-efficiency

```text
accuracy vs step-time ratio Pareto；
ValLossAUC_time paired bar；
forward/backward/step ratio stacked bar；
compact vs conservative memory bar；
time-to-target loss and accuracy；
classwise accuracy heatmap。
```

### 7.2 Geometry battery

```text
effective rank trajectory；
lift condition trajectory；
basis occupancy entropy trajectory；
curvature debt trajectory；
perturbation drift distribution；
local Jacobian proxy distribution；
CEp99 / MarginP10 trajectory；
ECE / NLL curve。
```

### 7.3 Signal / noise

```text
SNR positive fraction by role；
offdiag population proxy over training；
signal consistency real vs shuffled labels；
noise leak scatter；
role-wise update share stacked area。
```

### 7.4 Functional audit

```text
cos(d_new,d_task) heatmap；
predicted vs actual holdout descent scatter；
GeoCertificate delta bar；
control beat rate matrix；
lambda backtracking histogram；
bad-step timeline。
```

### 7.5 Final proof

```text
paired seed delta with CI；
method x dataset gate heatmap；
Pareto frontier: task vs geometry vs cost；
MLP analog / QuadraticFeatureMLP analog comparison；
failure taxonomy heatmap。
```

---

## 8. 失败后的决策规则

### Case A：base 不稳定

如果 repaired LQ 不能保持 robust near-pass 或 system envelope，则停止 functional update。

下一步只能做：

```text
global base repair；
lift scaling；
output scale；
initialization；
capacity envelope；
protocol mismatch audit。
```

不能做：

```text
functional 来救 base；
按数据集调 base；
打开 Conv / Former。
```

### Case B：LQ vs MLP 没有 geometry advantage

这不等于项目失败，但说明 KAN base contribution 只在表达/interaction 或效率方面成立，functional update 必须承担 geometry 改善。继续 Phase 3，但 final claim 必须写成 functional contribution，而不是 KAN geometry contribution。

### Case C：geometry metrics 不能预测任何任务或泛化指标

说明 Good Geometry Battery 还没定义对。下一步增加：

```text
perturbation directions；
tail strata；
leave-dataset-out；
leave-stratum-out；
train-stream probe；
noise-label split。
```

不要直接把某个 metric 当 loss。

### Case D：functional direction 不 task-safe

如果 holdout descent ratio、bad step rate 或 cos gate 不过，则 functional 不能进入训练。下一步只允许：

```text
减小 lambda；
projection onto task-safe subspace；
SNR / offdiag veto；
backtracking；
更局部的 geometry correction。
```

不能：

```text
降低 holdout gate；
删除 bad rows；
按 dataset 单独调 lambda。
```

### Case E：functional 被 controls 解释

如果 Random / AdamWParallel / SNR-only / MLP analog 也能得到同样收益，则不能 claim functional advantage。

下一步：

```text
重设 geometry target；
寻找 KAN-specific cover / basis / lift direction；
增加 control-gap objective；
缩小到 one-step/five-step audit，不跑 full training。
```

### Case F：收益只在一个数据集成立

可以作为 failure slice 记录，但不能 official promotion。下一步做：

```text
leave-dataset-out；
leave-stratum-out；
按 tail / rank / geometry condition 重新分层；
不要按 dataset_name 分支。
```

---

## 9. 最终判断

当前项目最本质的问题不是“functional update 为什么还没赢”，而是：

$$
\boxed{
\text{我们必须先让“好几何”成为可测、可证伪、可与 task / cost 同时成立的 certificate。}
}
$$

LQ / repaired LQ 已经给了我们第一次真正接近 MLP envelope 的 strict FC-PureKAN base。下一步必须围绕它做：

```text
1. 被动 Good Geometry Battery；
2. GeometryCertificateV0；
3. LQ vs MLP 的 KAN base contribution；
4. functional geometry maintenance 的 one-step / five-step control-resistant audit；
5. short-run / 10-seed confirmation。
```

这不是小修小补，而是把项目从“动作考古”和“future controller”拉回本质：

$$
\boxed{
\text{KAN 负责表达与 interaction cover；functional update 负责维护训练几何；controls 负责证明这不是错觉。}
}
$$
