# DG-KAN v13.0 Strategic Reset：Basis Substrate + Function-Preserving Coordinate Transport

> 版本：v13.0 strategic reset plan  
> 目的：停止继续扩同类 B-RAT / B-FOU / M-J token 网格；把研究从“候选小修”重置为两个核心问题：  
> 1. 经典基函数能否成为高效 substrate；  
> 2. functional update 是否应该改成 loss-agnostic 的 function-preserving coordinate transport，而不是一次性 output perturbation。  
> 公式格式：Typora 友好，只使用 `$...$` 和 `$$...$$`。  
> 硬约束：strict FC-PureKAN；不使用 label-informed initialization；functional direction 不使用 label、CE vector、permuted-label CE、validation/test、future outcome、query batch 或 dataset-name branch；CE / NLL / ECE / CEp99 只作为审计和坏化约束，不作为方向源。

---

# 0. 项目总目标与当前真实进展

## 0.1 项目总目标

DG-KAN 的目标不是在 MNIST / Fashion-MNIST / KMNIST 上打榜，也不是让某个 diagnostic score 好看。最终要证明的是：

$$
\boxed{
\text{label-free strict FC-PureKAN substrate + loss-agnostic functional update}
>
\text{same substrate + ordinary backprop / AdamW controls}
}
$$

并且必须同时满足：

```text
1. 表达力不打折；
2. forward / backward / step / memory 接近 MLP；
3. 训练轨迹健康，AUC-step / AUC-time 不输；
4. 几何健康，train-probe coupling / signal-reservoir-noise / tail / calibration 不坏；
5. functional update 的收益必须击败 NoOp / RandomMatchedNorm / AdamWParallel / SNR-only / MLP analog controls；
6. 不针对 CE 设计，不使用 label / validation / future / dataset-name branch。
```

## 0.2 当前真实进展

当前项目不再处于“完全没有 base / basis”的早期阶段。已经完成的事实是：

```text
1. 历史 FHQ/B320-current 证明过 strict FC-PureKAN 可以接近甚至优于 MLP-like efficiency envelope。
2. 但 B320-current 使用 label-informed trainprobe init，因此不能作为 official label-free claim。
3. all-basis no-BSpline map 已经执行，Rational 是目前唯一稳定 substrate family。
4. Non-RAT family 不是完全没 kernel；部分 lifetime / exact kernel 有实证，但 task-health 没有成立。
5. MLP functional closure 多轮为 0 pass，当前 generic MLP loss-agnostic observable family 不成立。
6. basis-specific functional P3 / P4 仍为 0 pass。
```

所以当前不是“再加一个小候选”的阶段，而是进入了核心机制问题：

$$
\boxed{
\text{我们有一些 substrate，但没有能改变 basis-channel 几何的 functional mechanism。}
}
$$

---

# 1. 为什么 v12.35 之后必须战略重置

## 1.1 旧路线的真实问题

v12.34.2 / v12.35 反复出现同一结构：

```text
有 substrate；
没有 healthy base；
没有 response dictionary pass；
basis-specific P3/P4 = 0；
MLP functional = 0；
Non-RAT task-health = 0。
```

继续扩：

```text
B-RAT6 / B-RAT7 / B-RAT8
B-FOU6 / B-FOU7
M-J4 / M-J5
```

大概率只是低价值排列组合，因为这些候选共享同一个失败机制：它们只是在 output / role / token 层移动，没有真正改变 basis 的训练坐标系统。

## 1.2 需要换的不是 gate，而是 functional update 的定义

过去的 functional update 更像：

$$
\theta' = \theta + \lambda d_{func}
$$

然后看它是否提高 CouplingR2、降低 NoiseSignalLeak、降低 ReservoirRatio。这个方向已经证明很难，因为当前 $d_{func}$ 没有可靠作用到 signal/reservoir/noise channel。

新的定义应该是：

$$
\boxed{
\text{functional update = function-preserving coordinate transport + future training geometry improvement}
}
$$

也就是说，functional update 不应该首先试图改变预测结果，而应该先改良模型的内部函数坐标，同时尽量保持当前函数不变。

---

# 2. 新核心假设

## H1：Basis 首要任务是成为 efficient substrate，而不是自己完全健康学习

每个 active basis family 的第一目标是：

```text
高效；
表达力有最低保障；
训练不灾难；
能暴露可操作的 basis-channel telemetry。
```

不要求 basis-only 先独立通过所有 AUC / LineC / tail gates。那些可以交给 functional transport 修复。否则 functional update 永远没有进入机会。

## H2：Functional update 应该做 function-preserving coordinate transport

给定一个已训练到中途的模型 $f_\theta$ 和无标签 train-stream batch $X$，先记录当前 logits：

$$
U = f_\theta(X).
$$

对 basis 参数做一个候选坐标变换 $T$，得到新 basis 表示 $\Phi_T(X)$。然后重新求解或补偿 readout / mixing，使输出尽量保持不变：

$$
W_T^* = \arg\min_W \|\Phi_T(X)W - U\|_F^2 + \lambda\|W-W_0\|_F^2.
$$

如果是多层内部 transport，则做局部函数保持：

$$
H_{\ell,T} W_T \approx H_{\ell} W_0.
$$

接受条件不是 CE 下降，而是：

$$
\frac{\|f_{\theta_T}(X)-f_\theta(X)\|_F}{\|f_\theta(X)\|_F+\epsilon}\le \epsilon_f,
$$

同时 basis-channel geometry 变好。

## H3：好 functional update 的收益应体现在未来训练轨迹，而不是 one-step loss

Coordinate transport 可能不会立刻提高 accuracy，但它应该让未来几百步 AdamW / manual AdamW 更健康：

```text
AUC-time 更好；
LineC 更稳；
NoiseSignalLeak 不升；
ReservoirRatio 不升；
CEp99 / NLL / ECE 不坏；
control gap 不能被 random transport / readout-only refit 解释。
```

因此 v13.0 不再要求 transport 事件本身造成即时 task gain。它要求：

$$
\boxed{
\text{function-preserving event 后的 future training trajectory 更好。}
}
$$

## H4：MLP 也必须有 analog control

如果 coordinate transport 在 MLP 上也有效，说明它可能是通用 reparameterization / optimizer mechanism；如果只在 KAN / Rational / Fourier 等 basis 上有效，才可能是 basis-specific functional geometry advantage。

MLP analog 包括：

```text
hidden feature whitening + inverse readout compensation；
weight balancing；
orthogonal hidden rotation + readout inverse rotation；
activation covariance re-centering。
```

这些 analog 也必须 loss-agnostic。

---

# 3. 新实验总结构

v13.0 分成六条线。

```text
Line S：Substrate Gate v2
Line X：Synthetic / controlled mechanism proof
Line G：Function-Preserving Coordinate Transport
Line M：MLP analog transport control
Line C：Manifold-Channel audit only
Line D：Classic no-BSpline basis substrate engineering
Line Z：Decision / no-go / finalizer
```

---

# 4. Line S：Substrate Gate v2

## 4.1 目标

把 active basis 从“候选”降级或升级为三类：

```text
S0 WorkspaceOnly:
  只是跑得动，但任务或几何太坏，不能 functional。

S1 EfficientSubstrate:
  跑得动，有表达，训练不灾难，有可用 telemetry，可以进入 coordinate transport。

S2 HealthyBase:
  basis-only 已经 task / AUC / LineC / tail 接近 MLP。
```

## 4.2 Active families

```text
D-RAT: Rational
D-CHE: Chebyshev
D-FOU: Fourier
D-RBF: RBF / FastKAN
D-WAV: Wavelet
D-FHQ: label-free FHQ monitor
```

B-spline 不进入 active budget。

## 4.3 S1 EfficientSubstrate gate

S1 不再要求完整 healthy base，但必须防止把明显坏掉的 substrate 送入 functional。

$$
workspace\_raw\_ratio \le 1.10,
$$

$$
workspace\_incremental\_ratio \le 2.00,
$$

$$
step\_ratio \le 2.00,
$$

$$
mean\_delta\_vs\_MLP \ge -0.08,
$$

$$
worst\_delta\_vs\_MLP \ge -0.15,
$$

$$
AUCtime\_ratio\_vs\_MLP \le 2.50,
$$

$$
LineC\_pass\_rate \ge 0.20.
$$

这是 substrate gate，不是 final success gate。

## 4.4 S2 HealthyBase gate

$$
mean\_delta\_vs\_MLP \ge -0.005,
$$

$$
worst\_delta\_vs\_MLP \ge -0.015,
$$

$$
AUCtime\_ratio\_vs\_MLP \le 1.05,
$$

$$
LineC\_pass\_rate \ge 0.70,
$$

$$
CEp99\_delta\_vs\_MLP \le 0.05,
$$

$$
ECE\_delta\_vs\_MLP \le 0.02.
$$

S2 如果过，basis-only 可作为 base；如果只过 S1，则进入 Line G functional transport。

## 4.5 必须记录

```text
candidate_id
family
basis_type
exact_kernel_implemented
uses_label_init
uses_ce_direction
raw_memory_ratio
incremental_memory_ratio
step_ratio
forward_ratio
backward_ratio
update_ratio
expression_E1/E2/E6/E8
mean_delta_vs_MLP
worst_delta_vs_MLP
AUC_step_ratio
AUC_time_ratio
ECE_delta
NLL_delta
CEp99_delta
LineC_pass_rate
CouplingR2
NoiseSignalLeak
RealSignalReservoirRatio
basis_telemetry_available
telemetry_fields_available
S0_workspace_only
S1_efficient_substrate
S2_healthy_base
fail_reason
```

---

# 5. Line X：Synthetic / controlled mechanism proof

## 5.1 目标

在真实数据上继续盲试 functional 很低效。v13.0 先建立可控机制证明：如果 functional transport 在已知几何任务上都不能改善未来训练，那就不该在 MNIST-like 任务上继续烧预算。

## 5.2 Synthetic tasks

```text
X1 Pairwise product
X2 Rotated quadratic
X3 Random quadratic
X4 Low-frequency + high-frequency mixture
X5 Local bump / tail cluster
X6 Two-manifold class split
X7 Label noise injection audit
```

这些任务不是打榜，而是机制单元测试。

## 5.3 对每个 basis 测什么

```text
1. basis-only training 是否能拟合；
2. transport 前后函数是否保持；
3. tangent condition 是否改善；
4. 继续训练 100 / 500 steps 后是否更快；
5. random matched transport 是否也有效。
```

## 5.4 通过标准

$$
FunctionDrift \le 0.01,
$$

$$
TangentCondition_{after} \le 0.7 \cdot TangentCondition_{before},
$$

$$
LossAUC_{after+train} \le 0.95 \cdot LossAUC_{noop+train},
$$

并且优于 random matched transport。

如果 synthetic 都不过，则该 basis 的 functional transport 暂停，不进入真实数据。

---

# 6. Line G：Function-Preserving Coordinate Transport

## 6.1 总体形式

对 substrate checkpoint $\theta_t$，使用无标签 batch $X_B$ 和 probe batch $X_Q$。

记录当前函数：

$$
U_B=f_{\theta_t}(X_B), \quad U_Q=f_{\theta_t}(X_Q).
$$

构造一个 basis-coordinate proposal $T$，得到 $\theta_T$，然后解补偿：

$$
\theta_{T,corr}=\operatorname{Compensate}(\theta_T, U_B).
$$

接受 transport 的第一道硬门是函数保持：

$$
Drift_B=\frac{\|f_{\theta_{T,corr}}(X_B)-U_B\|_F}{\|U_B\|_F+\epsilon}\le 0.01,
$$

$$
Drift_Q=\frac{\|f_{\theta_{T,corr}}(X_Q)-U_Q\|_F}{\|U_Q\|_F+\epsilon}\le 0.02.
$$

然后评估 coordinate geometry：

$$
GeomGain = z(ConditionDrop)+z(RankGain)+z(OccupancyGain)+z(TangentBalanceGain).
$$

只有函数保持 + 几何改善都成立，才跑 future training probe。

## 6.2 Family-specific coordinate transports

### Rational

```text
RAT-T1 denominator recenter + readout compensation
RAT-T2 slope cap / r' balancing + readout compensation
RAT-T3 tangent trust-region rescale
RAT-T4 group diversity orthogonalization
RAT-T5 readout-rational decoupling
```

Telemetry：

```text
den_p01, den_p99, den_condition
r_prime_p99, r_double_prime_p99
tangent_condition
tangent_top_eigen_share
group_function_diversity
readout_rational_coupling
```

### Chebyshev

```text
CHE-T1 degree-energy damping + coefficient compensation
CHE-T2 high-degree late-enable transport
CHE-T3 degree orthogonalization / recurrence rescale
```

Telemetry：

```text
degree_energy_k
high_degree_energy_ratio
recurrence_max_abs
degree_tangent_condition
```

### Fourier

```text
FOU-T1 band-energy rescale + readout compensation
FOU-T2 phase-stable low-frequency anchor
FOU-T3 high-frequency quarantine / late-enable
```

Telemetry：

```text
band_energy
high_freq_energy_ratio
phase_drift
frequency_tangent_condition
```

### RBF / FastKAN

```text
RBF-T1 center occupancy rebalance
RBF-T2 width condition rescale
RBF-T3 out-of-grid boundary recenter
RBF-T4 local bump merge / split
```

Telemetry：

```text
center_occupancy_entropy
dead_center_fraction
width_p01 / width_p99
out_of_grid_fraction
center_tangent_condition
```

### Wavelet

```text
WAV-T1 scale energy balance
WAV-T2 support overlap guard
WAV-T3 local-tail coverage repair
```

Telemetry：

```text
scale_energy
support_overlap
active_support_count
local_tail_coverage
scale_tangent_condition
```

## 6.3 Future training probe

Transport 不是即时 loss step。每个 accepted transport 都要比较未来训练：

```text
NoOp + train K steps
RandomMatchedTransport + train K steps
ReadoutOnlyRefit + train K steps
AdamWExtraStepMatchedTime + train K steps
FamilyTransport + train K steps
MLPAnalogTransport + train K steps
```

其中：

```text
K = 50, 200, 1000
```

## 6.4 Functional transport success gate

$$
FutureAUCtime_{transport} \le FutureAUCtime_{NoOp} - \delta_A,
$$

$$
LineC_{transport} \ge LineC_{NoOp},
$$

$$
NoiseSignalLeak_{transport} \le NoiseSignalLeak_{NoOp}+0.005,
$$

$$
ReservoirRatio_{transport} \le ReservoirRatio_{NoOp}+0.005,
$$

$$
CEp99_{transport} \le CEp99_{NoOp}+0.05,
$$

$$
ControlGap_{transport} \ge 0.005.
$$

---

# 7. Line M：MLP analog control

## 7.1 目的

验证 functional transport 是否 KAN-specific。

## 7.2 MLP analog candidates

```text
MLP-T1 hidden orthogonal rotation + inverse readout
MLP-T2 hidden whitening + readout ridge compensation
MLP-T3 layerwise weight balancing
MLP-T4 activation covariance re-centering
```

## 7.3 判断

如果 MLP analog 同样有效：

```text
functional transport 是通用 reparameterization mechanism；
需要和 KAN claim 分开写。
```

如果 MLP analog 无效，而 basis transport 有效：

```text
basis-specific functional geometry advantage 才成立。
```

如果两者都无效：

```text
当前 functional transport 机制 no-go；预算应回到 substrate / basis design。
```

---

# 8. Line C：Manifold-Channel audit only

Line C 继续记录，但不作为 direction source。

## 8.1 必须记录

```text
CouplingR2
CouplingCorr
NoiseSignalLeak
RealSignalReservoirRatio
KernelDrift
signal_effective_rank
reservoir_fraction
CEp99
NLL
ECE
Brier
margin_p10
```

## 8.2 解释规则

```text
LineC 变好 + function drift 小 + future training 更好：可疑似好 transport。
LineC 变好但 future training 不好：metric non-causal。
Future training 好但 LineC 坏：task-only trick，不能作为 geometry success。
Both 不好：candidate fail。
```

---

# 9. Line Z：Stop / go 规则

## 9.1 不再允许的路线

```text
1. 继续扩 B-RAT/B-FOU/M-J 同类 token 网格。
2. 只看 P3 one-step response 就 promotion。
3. 只因为 substrate workspace pass 就进入 official functional。
4. 用 CEp99 / NLL / ECE 构造方向。
5. 用 label / validation / future / dataset-name branch。
6. 把 diagnostic / smoke / skipped row 写成 success。
```

## 9.2 Go 条件

进入真实数据 functional P4 必须满足：

```text
1. S1 EfficientSubstrate pass；
2. Synthetic mechanism proof pass；
3. function-preserving transport pass；
4. future training probe pass；
5. controls pass；
6. provenance pass。
```

## 9.3 No-go 条件

如果某 family 满足：

```text
synthetic mechanism proof fail；
或 function-preserving transport impossible；
或 future training probe repeatedly no benefit；
或 MLP analog explains all benefit；
```

则该 family 的 functional transport 暂停。

---

# 10. 必须输出的 artifacts

```text
v130_substrate_gate_v2.csv
v130_synthetic_mechanism_proof.csv
v130_transport_proposals.csv
v130_function_preservation.csv
v130_basis_telemetry_before_after.csv
v130_future_training_probe.csv
v130_transport_controls.csv
v130_mlp_analog_transport.csv
v130_linec_audit.csv
v130_family_decision.csv
v130_route_decision.json
v130_failure_table.csv
v130_next_hypothesis_queue.md
v130_code_review_packet.zip
```

---

# 11. 必须生成的图

```text
fig_substrate_map_by_family.svg
fig_transport_function_drift_vs_geom_gain.svg
fig_future_auc_transport_vs_controls.svg
fig_basis_telemetry_before_after.svg
fig_linec_before_after_transport.svg
fig_synthetic_mechanism_success_matrix.svg
fig_mlp_analog_vs_basis_transport.svg
fig_family_go_nogo_dashboard.svg
```

---

# 12. Codex failure fallback

## 如果 function preservation 失败

Codex 必须先尝试：

```text
1. 增大 ridge compensation；
2. block-wise readout compensation；
3. smaller transport step；
4. solve readout + scale jointly；
5. local hidden reconstruction instead of final logits only。
```

## 如果 geometry 不改善

Codex 必须先尝试：

```text
1. 换 family-specific telemetry target；
2. 用 tangent condition 而不是 occupancy；
3. 用 occupancy 而不是 tangent condition；
4. 检查 telemetry 是否不敏感；
5. 标记 telemetry-no-go。
```

## 如果 future training 不改善

Codex 必须先尝试：

```text
1. K=50 / 200 / 1000 step sensitivity；
2. optimizer state transport vs reset；
3. event timing early/mid/late；
4. compare MLP analog；
5. compare readout-only refit。
```

## 如果 MLP analog 同样改善

Codex 必须输出：

```text
generic_reparameterization_effect = 1
KAN_specific_claim_allowed = 0
```

## 如果 all families fail synthetic proof

Codex 必须停止 functional transport，回到 substrate design，不允许继续扩 token。

---

# 13. 本版计划的核心变化

v13.0 与 v12.35 最大区别：

```text
旧路线：
  substrate -> response dictionary -> functional token -> P3/P4

新路线：
  substrate -> synthetic mechanism proof -> function-preserving coordinate transport -> future training probe -> P3/P4
```

这不是小修。它把 functional update 从一次性 perturbation 改成了 loss-agnostic coordinate surgery。

最终目标不变：

$$
\boxed{
\text{找到一个 label-free efficient PureKAN substrate，}
\text{并证明 functional transport 在该 substrate 上产生独立、control-resistant 的几何与训练收益。}
}
$$
