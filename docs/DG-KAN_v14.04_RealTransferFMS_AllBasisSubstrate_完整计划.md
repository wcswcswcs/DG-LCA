# DG-KAN v14.4：Real-Transfer Functional Metric State + All-Basis Substrate Consolidation 完整计划

> 版本：v14.4 execution plan  
> 基于：v14.3 `FunctionalValueConstraint AllBasisSubstrate` 最新结果复盘；v14.2 / v13.7 / v13.8 / v13.09 / v13.11 结果；`A Theory of Generalization in Deep Learning`；`Deep Manifold Part 2`  
> 目标：不再把 functional update 做成局部 token / 单次几何扰动 / response dictionary / oracle target；在已经打开 S3 synthetic 与 S4 real-short-run 的基础上，直接攻克 real-transfer stability，形成可 promotion 的 functional update。  
> 公式格式：Typora 友好，只使用 `$...$` 和 `$$...$$`。  
> 硬约束：strict FC-PureKAN；no active B-spline；no teacher；no distillation；no sampler / class weight；no dataset-name branch；no label-informed initialization；functional direction 不使用 validation / test / future / query batch；LineC / CEp99 / NLL / ECE / Brier 只能作为 audit / gate，不能作为方向源；MLP positive 只能作为 generic optimizer/control，不能写成 KAN-specific success。

---

# 0. 项目总目标与当前进展

## 0.1 总目标

DG-KAN 的最终目标不是在 MNIST / Fashion-MNIST / KMNIST 上打榜，也不是找到一个只在 synthetic 上有效的 KAN trick。项目目标是：

$$
\boxed{
\text{label-free strict FC-PureKAN substrate/base}
+
\text{loss-interface-generic functional update}
>
\text{same base + ordinary AdamW/backprop controls}
}
$$

其中 functional update 必须是研究突破本身，而不是基函数失败后的补丁。它必须满足：

```text
1. 表达力不打折；
2. forward / backward / memory / step 与 MLP 可比；
3. task 与 AUC-time 不坏，最好更快；
4. geometry 更健康：真实 signal 进入 signal channel，noise 不泄漏进 signal channel，reservoir 中真实信号可释放；
5. functional gain 必须打过 NoOp / RandomMatchedNorm / AdamWParallel / SNR-only / same-active-fraction controls；
6. functional direction 必须 loss-interface-generic，不允许 CE-tail / LineC-target / validation / test / future / dataset-name branch。
```

## 0.2 当前最新状态

v14.3 的最新状态已经不是过去的 R4 no-go。当前最新 synthetic/all-basis artifact 达到：

```text
route = S3-KANSpecificSyntheticPass
minimum_success = S3-KANSpecificSyntheticPass
official_success_reached = 1
rational_fms_task_pass_count = 5
kan_specific_positive_rows = 12
nonrat_strict_substrate_pass_count = 3
real_short_run_open_allowed = 1
promotion_allowed = 0
```

这说明：**functional update 已经打开 KAN-specific synthetic S3**，并且 Non-RAT substrate 不再完全为 0。

最新 real short-run 也已合法打开并执行：

```text
route = S4-RealShortRunOpened
official_s5_reached = 0
real_dataset_seed_pass_count = 4 / 9
real_short_run_pass_rows = 4
mean_source_vs_best_control_noncontrol = 0.06567642423841688
promotion_allowed = 0
```

所以当前不是“functional 完全没进展”。更准确地说：

$$
\boxed{
\text{v14.3 已经从 S3 进入 S4；}
\text{现在卡在 real-transfer stability，即 4/9 不能推广到 9/9。}
}
$$

## 0.3 当前不能过度 claim 的地方

v14.3 仍没有达成：

```text
S5-OfficialFunctionalSuccess = 0
promotion_allowed = 0
real 3x3 pass = 4/9, not 9/9
```

也就是说：

```text
不能说 DG-KAN 已经成功；
不能说 functional update 已经 official；
不能说 real-short-run 已经稳；
不能把 4/9 positive 写成 promotion；
不能把 MLP / generic FMS positive 写成 KAN-specific success。
```

---

# 1. 各条线当前进展百分比

这些百分比不是 artifact 官方字段，而是根据 gate 通过情况、机制清晰度、代码完整性、离 official success 的距离综合判断。

| 线 | 当前完成度 | 判断 |
|---|---:|---|
| Line R：代码 / provenance / finalizer 审计 | 99% | 工程闭包强，不是当前 blocker |
| Historical FHQ / B320-current 工程能力 | 85% frozen | 历史强，但 label-informed init 已禁用，不能 official |
| Line C：Manifold-channel 几何审计 | 88% | 审计稳定，但不能作为方向源 |
| Generic PopRisk / FMS 实现面 | 90% | persistent state、per-example gradients、amortized update、projection trace 已成熟 |
| Generic MLP-FMS | 45% | generic signal 存在，但不能作为 KAN promotion |
| KAN-specific FMS synthetic | 55% | 已达 S3，rational_fms_task_pass_count=5 |
| KAN-specific real short-run | 35% | 已打开 S4，best real 3x3 = 4/9 |
| Rational substrate | 85% | 当前最稳定 substrate |
| Rational FMS functional | 45% | synthetic S3 成立，但 real S5 未成立 |
| Wavelet substrate | 45% | D-WAV17/18/19 robust substrate signal 出现，仍需 hardening |
| RBF / FastKAN substrate | 20% | task collapse 未解决 |
| Chebyshev substrate | 15% | exact/manual rows 有信号，但 full-step incremental memory 未闭合 |
| Fourier substrate | 20% | memory/step 局部好，但 strict substrate 仍未稳 |
| Non-RAT functional | 10% | 有 3 个 strict substrate pass，但还未进入 official FMS proof |
| 整体 next-gen MLP claim | 42%-50% | S3/S4 是真实进展，但 S5 仍为 0 |

---

# 2. 本轮结果独立分析

## 2.1 最大进展：S3 与 S4 已经打开

过去很多轮我们一直停在：

```text
response dictionary = 0
oracle = 0
one-shot SNR = 0
cover objective invalid
FMS generic only
Rational local rows rejected
```

v14.3 最新结果改变了状态：

```text
1. Rational/KAN synthetic S3 达成；
2. Non-RAT strict substrate pass_count 从 0 提升到 3；
3. real short-run 合法打开；
4. real 3x3 最好达到 4/9 pass；
5. mean_source_vs_best_control_noncontrol 为正。
```

这说明 functional update 不是完全错误的方向。它已经在 synthetic 与真实短训中产生可测量 value。

## 2.2 但 S3/S4 不是 S5

S5 失败的核心不是“没有 source”，而是：

```text
1. source instability：部分 dataset/seed source_vs_control 仍为负；
2. tail instability：CEp99 / NLL / ECE 在一些 seed 上坏化；
3. LineC instability：real LineC 不是所有 seed 都稳定；
4. time / update overhead：部分 strong source 被 step-time gate 卡住；
5. transfer gap：synthetic S3 到 real 3x3 的转移只有 4/9。
```

所以当前 blocker 已从：

```text
functional 有没有信号？
```

变成：

```text
functional signal 如何在 real train-stream 上稳定转移？
```

这是一种进展，但也是更硬的 blocker。

## 2.3 不应继续的旧路线

v14.3 已经覆盖了大量旧路线：

```text
Rational projection / K-method allk；
K8 real short-run；
K1/K2/K3 alternate；
strength 0.05；
longer 400-step；
batch64；
label-free train-entropy output geometry；
Non-RAT compact hidden repair；
manual no-materialize workspace；
Wavelet support repair；
lower raw residual repair；
role constraint repair；
output geometry repair；
support+output geometry combination；
RBF center/width repair；
RBF longer hardening。
```

这些路线给出的结论是：**继续扩 token 小网格已经不是主要路径。**

---

# 3. 结合文档后的本质理解

## 3.1 `A Theory of Generalization in Deep Learning` 的启发

这篇文档的关键不是“再加一个 SNR trick”，而是：训练信号应该由 per-example gradient 的 drift-vs-diffusion 决定。

对 batch $B$：

$$
\bar g_B=\frac{1}{b}\sum_i g_i,
$$

$$
\Sigma_B=\frac{1}{b}\sum_i(g_i-\bar g_B)(g_i-\bar g_B)^T,
$$

$$
A_B=\bar g_B\bar g_B^T-\frac{1}{b-1}\Sigma_B.
$$

对 diagonal 情况，population-safe gate 是：

$$
\mu_k^2>\frac{\sigma_k^2}{b-1}.
$$

这说明 functional update 的合法信息接口应该是：

```text
loss-interface-generic per-example gradients
+
train-stream population-risk statistics
```

而不是：

```text
validation/test/future；
LineC hard target；
CEp99/NLL/ECE direction；
dataset-name branch。
```

## 3.2 `Deep Manifold Part 2` 的启发

Deep Manifold 的关键启发是：网络不是固定坐标里的静态函数，而是 moving coordinates / shifting node covers / boundary-conditioned iteration。Node covers 在训练中不断移动，fixed-point regions 是训练过程构造出来的。

这说明 functional update 不应该是一次性 correction，而应该是训练过程中的 boundary / metric state：

```text
每步用 train-stream population-risk 信号决定 plasticity；
用 basis telemetry 约束哪些方向不能过度移动；
让 cover / chart 在训练中逐步稳定。
```

## 3.3 对当前结果的解释

v14.3 的结果说明：

```text
1. generic FMS 有 value；
2. Rational/KAN 可以在 synthetic 上把 value 放大到 S3；
3. real short-run 也能出现 4/9 positive；
4. 但 train-stream 的 value / risk / tail / LineC 没有在 real 3x3 中稳定对齐。
```

所以 v14.4 不应该再问：

```text
再哪个 K-token 会过？
```

而应该问：

$$
\boxed{
\text{怎样用 train-stream-only 约束，把 synthetic S3 的 value 转成 real 3x3 S5？}
}
$$

---

# 4. 当前核心假设

## H1：real S5 失败是 value-risk 不同位，不是 source 不存在

v14.3 已经有 strong source rows，也有 4/9 real pass。因此 blocker 不是 source completely absent。

假设：

```text
generic FMS value 可以产生 source；
basis constraint 可以保护部分 geometry；
但当前没有一个 train-stream-only risk state 同时约束 tail / LineC / transfer stability。
```

要验证：

```text
pre_projection_source_vs_control
post_projection_source_vs_control
train_stream_loss_tail_proxy
train_stream_logit_entropy
train_stream_margin_p10
train_stream_gradient_snr_tail
projection_rejection_fraction
source-tail co-location
```

## H2：S5 需要 real-transfer constraint，而不是 synthetic gate 再加严

Synthetic 5/7 已经打开，但 real 只有 4/9。继续加严 synthetic gate 可能只会筛掉 source，而不能解决 real transfer。

需要新增：

```text
real-transfer value constraint
```

该 constraint 只能使用 train-stream precommit 信息，例如：

```text
current train batch logits；
train-stream per-example loss distribution；
train-stream gradient SNR；
split-half source agreement；
FMS state stability；
basis projection rejection fraction；
logit RMS / entropy / margin proxies。
```

不能使用：

```text
validation/test；
future；
LineC hard target；
CEp99/NLL/ECE audit；
dataset-name branch。
```

## H3：Rational 是当前 functional carrier，Wavelet 是最接近的 Non-RAT carrier

Rational 已经是 S3 主力。Wavelet 在 v14.3 后续中出现了 robust substrate signal：

```text
D-WAV17 / D-WAV18 / D-WAV19
```

它们值得进入 substrate hardening，但不能直接写成 all-basis success。

## H4：Non-RAT 需要分 family 处理

当前不能把 Non-RAT 统称为失败：

```text
Wavelet:
  output geometry / train-entropy / low-raw residual 是真实线索。

RBF:
  LineC 有时可见，但 task collapse 是 blocker。

Chebyshev / Fourier:
  exact/manual path 降低 raw / step，但 incremental memory lifetime 仍是 blocker。
```

---

# 5. v14.4 总体执行结构

```text
Line R:
  implementation / provenance / route audit。

Line G:
  generic FMS value confirmation, low budget control。

Line K:
  Rational Real-Transfer Value Constraint。

Line W:
  Wavelet substrate hardening and optional FMS shadow。

Line D:
  RBF / Chebyshev / Fourier / Wavelet all-basis substrate repair。

Line C:
  LineC / tail / calibration audit only。

Line Z:
  finalizer / route / no-go / next queue。
```

预算建议：

```text
Line K Rational real-transfer FMS: 35%
Line W/D all-basis substrate: 35%
Line G MLP generic control: 10%
Line C/R/Z audit/finalizer: 20%
```

---

# 6. Line K：Rational Real-Transfer Value Constraint

## 6.1 目标

把 v14.3 的：

```text
synthetic S3 + real 4/9
```

推进到：

```text
real 3x3 S5
```

重点不是再找 source，而是让 source 与 tail / LineC / AUC / step-time 同位。

## 6.2 候选机制

### K-RT1：GenericFMS + TrainSplitAgreement

在每个 train-stream step，把当前 batch 分成两半 $B_1,B_2$，分别估计 FMS value direction：

$$
\Delta\theta_{FMS}^{(1)},\quad \Delta\theta_{FMS}^{(2)}.
$$

定义 agreement：

$$
A_t=
\cos(
\Delta\theta_{FMS}^{(1)},
\Delta\theta_{FMS}^{(2)}
).
$$

只有当：

$$
A_t \ge \tau_A
$$

才允许完整 FMS；否则 blend 回 AdamW：

$$
\Delta\theta_t =
\lambda_t \Delta\theta_{FMS}
+
(1-\lambda_t)\Delta\theta_{AdamW}.
$$

这里 $\lambda_t$ 由 agreement 连续调节，不是 hard threshold 小修：

$$
\lambda_t=\operatorname{clip}\left(\frac{A_t-a_0}{a_1-a_0},0,1\right).
$$

### K-RT2：GenericFMS + TrainStreamTailTrust

定义 train-stream per-example loss tail proxy：

$$
Q_{loss,t}=\operatorname{Quantile}_{0.95}(\ell_i).
$$

定义 margin proxy：

$$
m_{10,t}=\operatorname{Quantile}_{0.10}(p_{top1}-p_{top2}).
$$

定义 logit RMS proxy：

$$
R_{logit,t}=\sqrt{\mathbb E[\|z_i\|_2^2]}.
$$

这些都是 train-stream 当前 batch 可见，不是 validation/test audit。

用它们构造 trust state：

$$
q_{t+1}
=
\gamma q_t
+
(1-\gamma)
\left[
w_1 \Delta Q_{loss,t}
-
w_2 \Delta m_{10,t}
+
w_3 \Delta R_{logit,t}
\right].
$$

如果 trust state 恶化，则降低 FMS plasticity：

$$
\lambda_t=\sigma(-q_t).
$$

### K-RT3：GenericFMS + ProjectionValueRetention

从 v14.3 已有 projection audit 扩展：

$$
retention_t
=
\frac{
\|\Pi_{basis-safe}(\Delta\theta_{FMS})\|
}{
\|\Delta\theta_{FMS}\|+\epsilon
}.
$$

如果 projection 抹掉大部分 value，说明 basis constraint 正在杀死 generic FMS：

```text
retention too low -> reduce basis constraint strength；
retention high but tail bad -> strengthen basis safety；
retention high and safe -> allow FMS。
```

### K-RT4：GenericFMS + SourceTailCoState

同时维护两个 state：

$$
c^{value}_{t+1}
=
\beta c^{value}_t
+
(1-\beta)U_t
$$

$$
c^{risk}_{t+1}
=
\beta c^{risk}_t
+
(1-\beta)R_t
$$

update strength：

$$
\lambda_t
=
\operatorname{clip}
\left(
\frac{c^{value}_t}{|c^{risk}_t|+\epsilon},
0,
\lambda_{max}
\right).
$$

这个机制的目标是让 value 与 risk 同位，而不是单独最大化 value。

### K-RT5：GenericFMS + DelayedBasisConstraint

前期不强约束 basis：

```text
Phase 1:
  generic FMS + weak basis constraint。

Phase 2:
  projection retention audit + moderate constraint。

Phase 3:
  stronger constraint only if train-stream tail proxy stable。
```

这吸收 Deep Manifold 的 plasticity-rise-then-consolidate 思想。

## 6.3 必须记录

```text
dataset
seed
method
step
lambda_fms
train_split_agreement
train_loss_q95
train_margin_p10
train_logit_rms
value_state
risk_state
projection_value_retention
projection_rejection_fraction
cos_projected_vs_generic
active_fraction
update_norm
step_time_overhead
source_vs_best_control
source_vs_noop
AUCtime_ratio
LineC_pass
CouplingR2_delta
NoiseSignalLeak_delta
RealSignalReservoirRatio_delta
CEp99_delta
NLL_delta
ECE_delta
```

## 6.4 Gate

Real-transfer gate：

```text
real_dataset_seed_pass_count >= 6/9 for exploration；
real_dataset_seed_pass_count = 9/9 for S5 official。
```

Row-level pass：

$$
source\_vs\_best\_control \ge 0.005,
$$

$$
AUCtime\_ratio \le 1.0,
$$

$$
CEp99\_delta \le 0.05,
$$

$$
NLL\_delta \le 0.02,
$$

$$
ECE\_delta \le 0.02,
$$

$$
LineC\_pass = 1.
$$

No gate lowering. If 6/9 achieved but not 9/9, route is:

```text
S4b-RealTransferExplorationPositive
```

not S5.

---

# 7. Line W：Wavelet substrate hardening

## 7.1 目标

v14.3 显示 D-WAV17/18/19 是真实 Non-RAT substrate signal。Line W 不进入 official FMS proof，先做 substrate hardening。

## 7.2 候选

```text
W1-DWAV17-Raw005SupportHealth
W2-DWAV18-Raw002SupportHealth
W3-DWAV19-Raw001SupportHealth
W4-DWAV19-TrainEntropyOutputGeometry
W5-DWAV19-Support075TrainEntropy
W6-DWAV19-LowRawResidualDelayedReadout
W7-DWAV19-ScaleSupportOccupancyBalance
```

## 7.3 关键原则

Wavelet repair 只能使用：

```text
train-stream logits；
train-stream x support stats；
train-stream entropy；
basis support occupancy；
scale energy；
logit RMS。
```

不能使用：

```text
LineC / CEp99 / NLL / ECE / validation / test / future
```

作为方向或 substrate rule。

## 7.4 Gate

Wavelet substrate gate：

$$
workspace\_raw\_ratio \le 1.10,
$$

$$
workspace\_incremental\_ratio \le 1.75,
$$

$$
step\_ratio \le 1.75,
$$

$$
mean\_delta\_vs\_MLP \ge -0.05,
$$

$$
worst\_delta\_vs\_MLP \ge -0.10,
$$

$$
AUCtime\_ratio \le 2.0,
$$

$$
LineC\_pass\_rate \ge 0.30.
$$

Hardening success：

```text
3 datasets x 3 seeds all pass substrate gate；
or at least 8/9 pass + no dataset systematic collapse。
```

---

# 8. Line D：All-basis substrate repair

## 8.1 RBF / FastKAN

Current blocker：

```text
task collapse despite occasional LineC signal。
```

Next directions：

```text
D-RBF-A active center occupancy repair
D-RBF-B width quantile + OOG guard
D-RBF-C compact h256 no-materialize with residual identity
D-RBF-D triangular/RBF hybrid FastKAN
D-RBF-E center dropout / occupancy balance
```

Metrics：

```text
center_occupancy_entropy
width_condition
out_of_grid_fraction
active_center_fraction
task mean/worst/AUC
LineC pass
workspace ratios
```

## 8.2 Chebyshev

Current blocker：

```text
manual/exact rows reduce raw/step cost, but incremental memory remains high。
```

Next directions：

```text
D-CHE-A degree recurrence recompute
D-CHE-B delayed optimizer-state allocation
D-CHE-C degree-energy cap
D-CHE-D low-degree identity residual
D-CHE-E degree-wise FMS shadow only after substrate gate
```

Metrics：

```text
degree_energy_k
high_degree_ratio
incremental_memory_ratio
readout_grad_live_bytes
coeff_grad_live_bytes
workspace_lifetime_waterfall
```

## 8.3 Fourier

Current blocker：

```text
low-frequency path memory/step local signals exist; task/expression/LineC not stable。
```

Next directions：

```text
D-FOU-A band-limited K2/K4 with identity residual
D-FOU-B phase-amplitude shared no-materialize
D-FOU-C high-frequency quarantine
D-FOU-D delayed band expansion
D-FOU-E frequency-band occupancy audit
```

## 8.4 Wavelet

Handled in Line W as the main Non-RAT candidate.

## 8.5 Rational

Rational remains main functional carrier. Substrate repair only if v14.4 reveals structural regression.

---

# 9. Line G：Generic MLP-FMS control

MLP-FMS remains important but cannot be KAN promotion.

Run:

```text
MLP-AdamW
MLP-FMS-best-v14.2
MLP-FMS-Amortized
MLP-FMS-SourceTailCoState
MLP-FMS-SplitAgreement
```

Purpose:

```text
1. confirm generic FMS value source；
2. compute KAN-specific advantage:
   (Rational-FMS - Rational-AdamW) - (MLP-FMS - MLP-AdamW)；
3. prevent writing generic optimizer as KAN-specific success。
```

If MLP-FMS succeeds and KAN does not:

```text
Generic FMS works; KAN-specific advantage not established。
```

---

# 10. Line C：Manifold-channel audit

Line C remains audit-only.

Record:

```text
CouplingR2
CouplingCorr
NoiseSignalLeak
RealSignalReservoirRatio
signal_mass_topk
reservoir_fraction
kernel_drift
CEp99
NLL
ECE
Brier
margin_p10
```

No candidate may use LineC to generate direction.

Visualizations:

```text
fig_real_3x3_pass_matrix.svg
fig_source_vs_tail_scatter.svg
fig_projection_retention_vs_source.svg
fig_train_stream_proxy_vs_real_audit.svg
fig_linec_failure_by_dataset_seed.svg
fig_basis_substrate_matrix.svg
```

---

# 11. Final route definitions

```text
R0-ImplementationOrProvenanceFail
R1-NoS3Regression
R2-RealTransferFail
R3-NonRATSubstrateFail
S3-KANSpecificSyntheticPass
S4-RealShortRunOpened
S4b-RealTransferExplorationPositive
S5-OfficialFunctionalSuccess
```

## S5 official success

S5 requires:

```text
1. S3-KANSpecificSyntheticPass remains true；
2. real short-run 3x3 = 9/9 dataset-seed pass；
3. source_vs_best_control >= 0.005 in all rows；
4. AUCtime_ratio <= 1.0；
5. CEp99/NLL/ECE non-harm；
6. LineC pass；
7. no forbidden direction source；
8. promotion_allowed can be set to 1 only after manifest / provenance / code review pass。
```

---

# 12. Codex failure-response rules

If real pass count remains 4/9:

```text
Do not lower gate.
Do not branch by dataset.
Run failure decomposition:
  source instability vs tail instability vs LineC instability vs overhead.
Then try K-RT1..K-RT5.
```

If source improves but tail fails:

```text
Try train-stream tail trust / risk state.
Do not use CEp99 audit as direction.
```

If tail improves but source fails:

```text
Try projection value retention / split agreement.
Do not switch to validation selection.
```

If LineC fails:

```text
Audit role/basis movement.
Do not use LineC target direction.
Try delayed basis constraint or weaker constraint phase.
```

If Wavelet substrate fails seed1/2:

```text
Try train-stream entropy / support occupancy / output RMS rule.
Do not use seed-specific scaling.
```

If RBF task collapses:

```text
Try center occupancy / width condition / identity residual.
Do not run official FMS proof.
```

If Non-RAT remains fail:

```text
Do not block Rational S5.
Keep Line D parallel with bounded budget.
```

---

# 13. 必须输出 artifacts

```text
v144_route_decision.json
v144_required_manifest.csv
v144_forbidden_information_audit.csv
v144_code_review_manifest.csv

Line K:
v144_real_transfer_fms_results.csv
v144_real_transfer_fms_summary.csv
v144_train_stream_proxy.csv
v144_projection_value_retention.csv
v144_source_tail_costate.csv
v144_split_agreement.csv
v144_real_3x3_failure_table.csv

Line W/D:
v144_wavelet_substrate_hardening.csv
v144_rbf_substrate_repair.csv
v144_chebyshev_lifetime_repair.csv
v144_fourier_lifetime_repair.csv
v144_all_basis_substrate_status.csv

Line G:
v144_mlp_generic_fms_control.csv
v144_kan_specific_advantage.csv

Line C:
v144_linec_audit.csv
v144_tail_calibration_audit.csv

Figures:
fig_real_3x3_pass_matrix.svg
fig_source_vs_tail_scatter.svg
fig_projection_retention_vs_source.svg
fig_train_stream_proxy_vs_real_audit.svg
fig_linec_failure_by_dataset_seed.svg
fig_basis_substrate_matrix.svg
fig_kan_specific_advantage.svg
```

---

# 14. 最终判断

v14.3 的意义是：**functional update 已经不是空想，它已经达到 S3 并打开 S4。**

但 v14.3 也证明：**synthetic S3 不等于 real S5。**

现在真正的科学问题是：

$$
\boxed{
\text{怎样用 train-stream-only 的 value-risk boundary，}
\text{把 generic FMS / Rational FMS 的 source 转成 real 3x3 robust gain？}
}
$$

v14.4 不是回到小修，也不是回到 basis-only。它的核心是：

```text
functional 继续作为突破口；
Rational 作为主 carrier；
Wavelet / RBF / Chebyshev / Fourier 作为并行 substrate line；
MLP 作为 generic FMS control；
real-transfer stability 作为主 gate。
```

如果 v14.4 不能把 4/9 推到至少 6/9，说明当前 real-transfer functional mechanism 仍不够；如果能推到 6/9 但不到 9/9，继续研究 real-transfer boundary；如果达到 9/9，才允许进入 S5 official functional success。
