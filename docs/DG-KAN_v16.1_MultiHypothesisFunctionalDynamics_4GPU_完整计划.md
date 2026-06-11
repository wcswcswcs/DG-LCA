# DG-KAN v16.1：Multi-Hypothesis Functional Dynamics + MLP Active + LQ Reanchor + All-Basis + 4GPU 并行加速完整计划

> 版本：v16.1 execution plan  
> 生成时间：2026-06-01  
> 依据：v16.0 真实结果复盘、v16.0 修正版计划、v15.4-v15.9 split-consensus / dynamics 结果、v12.5.1 Manifold-Channel Geometry 诊断、Deep Manifold / Generalization 文档启发  
> 公式格式：Typora 友好，只使用 `$...$` 与 `$$...$$`  
> 硬约束：strict FC-PureKAN；no active B-spline；no teacher；no distillation；no loss modification；no sampler / class weight；no dataset-name branch；no seed-specific scale；no fake / proxy / CPU offload；functional direction 不使用 validation / test / future / query；LineC / CEp99 / NLL / ECE / AUCtime / Brier 只能作为 audit / gate / debt readback，不能作为方向源。  
> 核心修正：v16.0 做到了多 carrier，但每个 carrier 的 functional 方案仍偏少；v16.1 必须把 **carrier 维度** 和 **functional 机制假设维度** 同时展开，用四张 GPU 并行裁决多种互斥假设，不能再等一个实验失败后写下一版。

---

# 0. 项目总目标与当前真实状态

DG-KAN 的总目标仍然是：

$$
\boxed{
\text{在 strict FC-PureKAN base 上，通过 functional update 改善训练动力学，}
\text{最终得到比普通 AdamW/backprop 更好的模型。}
}
$$

这里的“更好”不是局部指标变好，也不是 one-step source 变正，而是同时满足：

```text
1. base / substrate 效率接近 MLP；
2. functional update 的收益超过 AdamW / random / matched / generic optimizer controls；
3. 训练轨迹长期更好，而不是只看 single-step source；
4. 短期 tail / LineC / calibration / AUC debt 可以被后续训练动力学偿还；
5. 若 MLP-FU 也成功，必须写成 generic training-dynamics insight，不能写成 KAN-specific；
6. 若 KAN carrier 比 MLP 更强，才可以讨论 KAN-specific functional advantage；
7. 所有 positive-looking rows 都必须通过 matched controls、MLP controls、NoOp overhead controls 和 generic optimizer controls。
```

v16.0 已经修正了一个重大执行问题：`h400` fail 不再自动阻断 `h800/h1600`。A/B full surface 已执行到 H=800，top-2 source-retaining candidates 已执行 H=1600。结果显示：当前 D-CHE / MLP / LQ / Rational / all-basis 组合没有形成 productive debt recovery，当前 route 是 `R16-CurrentFunctionalDynamicsFamilyNoGo`。这不是工程漏跑，而是当前预注册 functional dynamics family 的真实 no-go。

v16.1 的关键不是继续 v16.0 的一个方案，而是把 functional update 设计成 **多机制假设矩阵**：

$$
\boxed{
\text{carrier 多线并行}
\times
\text{functional 机制多假设并行}
\times
\text{长期 dynamics horizon 并行}
}
$$

---

# 1. 各条线当前进展百分比与独立判断

| 线 | 当前完成度 | 判断 |
|---|---:|---|
| 代码 / provenance / no-action audit | 99% | 工程闭包强；v16.0 required / forbidden / no-action / budget audit 已闭合。 |
| Historical B320 / FHQ anchor | 85% frozen | 历史强，但 label-informed init 禁用，不能作为 final official claim。 |
| D-CHE substrate | 85% | 当前最强 KAN carrier，历史 9/9 substrate eligibility。 |
| D-CHE split-consensus source | 45%-50% | G7R / A8 lineage 有 source 线索，但 h800/h1600 未保住。 |
| D-CHE dynamic recovery | 5%-10% | v16.0 A8 h800 有 source 与 LineC 局部恢复，但 source retention / tail / AUC 不成立；h1600 退化。 |
| MLP functional dynamics | 25%-35% | B7/B10 h800 source retention 与 AUC 比 D-CHE 更像有效动力学，但 tail/LineC 不偿还，且是 generic，不是 KAN-specific。 |
| LQ reanchor / late attach | 25%-35% | C0/C4/C6 有 6/9 near-pass 线索，但 reanchor gate = 0，不能直接开 functional。 |
| Rational substrate / monitor | 70%-80% substrate / 10% functional | 稳定 monitor；reset / optimizer-state route 被 generic confound 打回，不重启 controller/reset/action。 |
| D-FOU substrate | 20%-30% | 历史 6/9，v16.0 仍 0/9；需要 carrier/substrate repair，不得直接 official FU proof。 |
| D-RBF / FastKAN substrate | 20%-30% | workspace/telemetry 有线索，task-health 不稳。 |
| D-WAV substrate | 15%-20% | 弱线索，低预算保留。 |
| Dynamic geometry definition | 20%-25% | v16.0 证明“短期坏债可偿还”在当前定义下不成立；但也暴露 DGS 需要重定义。 |
| Recovery mechanisms | 15%-20% | AdamW / cooldown / SWA / decay 等已部分测过，但没有系统比较多机制 × 多 carrier × 长 horizon。 |
| Multi-hypothesis FU matrix | 0% | v16.1 新主线：每个 carrier 同时跑多种 functional 机制，而不是单方案等待。 |
| Official S5 | 0% | 尚未达成。 |
| 整体 next-gen MLP claim | 23%-31% | 科学 claim 下调；工程成熟，但 current train-stream FU family no-go，需要机制矩阵重开。 |

---

# 2. v16.0 的独立分析：有进展，但不是能力进展

v16.0 的价值在于排除了一个重要反方：不是因为 `h400` 太早挡住长期恢复。A/B 已执行到 h800，并对 top-2 做 h1600 extension。

最接近希望的 D-CHE row 是：

```text
A8-D-CHE-G7R-PulseOnce-then-LRCooldownRecovery
source_vs_best_control_h800 = 0.049836417039235435
source_retention_h800 = 0.24507864895794126
LineC_recovery_rate_h800 = 0.5555555555555556
tail_recovery_rate_h800 = 0.023526686156257203
AUCtime_ratio_h800 = 1.1013260447040822
```

这说明 A8 不是完全无信号，但不能叫 productive dynamics：source 没保住，tail debt 没偿还，AUCtime 还变慢。MLP active line 反而给出更强 generic 动力学信号：

```text
B7-MLP-PulseOnce-then-LRCooldownRecovery
source_vs_best_control_h800 = 0.06957992580201891
source_retention_h800 = 0.648903876543045
AUCtime_ratio_h800 = 0.9137877291838247
LineC_recovery_rate_h800 = 0.0
tail_recovery_rate_h800 ≈ 0.001
```

MLP B7/B10 说明 generic training dynamics 可能有 source-retention / AUC 线索，但它没有偿还 tail/LineC，而且不是 KAN-specific。Line M 中 13 个 KAN-vs-MLP attribution rows 的 KAN-specific advantage 是 0/13，best D-CHE A8 相对 best MLP B7 的 `delta_KAN_specific = -0.019743508762783475`。h1600 进一步关闭“再等更久就翻盘”的解释：A8 到 h1600 后 mean source 约为负，B7/B10 也变负且 bad event 仍为 1.0。

因此，v16.0 的真实结论是：

$$
\boxed{
\text{当前单一路线的 pulse+recovery dynamics 没有形成 productive debt repayment。}
}
$$

但 v16.0 暴露了另一个问题：每条 carrier 只跑一类主方案，收集到的机制线索仍太少。v16.1 必须一次性并行验证多套 functional update 假设。

---

# 3. 当前问题的本质

当前 blocker 不是单个指标 fail，也不是 Codex 没继续。当前 blocker 是：

$$
\boxed{
\text{source retention、tail/LineC debt、control-equivalence 三者耦合失败；}
\text{当前方案没有把它们拆开，也没有解释长期动力学为什么不能偿还。}
}
$$

具体表现：

```text
1. D-CHE 有一点 LineC recovery，但 source retention / tail recovery / AUC 不成立。
2. MLP 有 source-retention / AUC 线索，但 tail / LineC debt 不偿还，且是 generic。
3. LQ 有 near-pass base 线索，但 reanchor 未闭合。
4. Rational 只能 monitor，不能重启 reset route。
5. D-FOU / D-RBF / D-WAV 仍没给出 carrier escape。
6. DGS 没能解释为什么 B7/B10 source-retention dynamics 不偿还 LineC/tail。
```

所以下一步不能继续：

```text
G9/G10；
FU9/FU10；
action bank / controller；
reset route；
trust scalar 网格；
按 dataset / seed fail pattern 写 branch；
用 LineC / CEp99 / NLL / ECE / AUCtime 反推方向。
```

下一步必须：

```text
1. 多 carrier 同时跑；
2. 每个 carrier 不只跑一种 FU，而是多机制矩阵；
3. 所有机制共享统一 horizon/debt/controls；
4. 用 4 GPU 明确并行，不再串行等一条线失败；
5. 每轮结束必须给出机制裁决，而不是只给 route。
```

---

# 4. v16.1 的核心原则：carrier × mechanism 矩阵

v16.1 不再写成一条主线。它是一个矩阵：

$$
\boxed{
\mathcal E
=
\text{Carriers}
\times
\text{MechanismFamilies}
\times
\text{RecoveryHorizons}
\times
\text{Controls}
}
$$

## 4.1 Carrier 维度

```text
C1: D-CHE
  当前最强 KAN carrier；必须继续，但不能独占。

C2: MLP
  主动 functional dynamics 发现线，不再只是 control。

C3: LQ
  reanchor + late attach；不 reanchor 不进入 official functional。

C4: Rational
  monitor + generic dynamics sanity；不重启 reset/controller/action。

C5: D-FOU
  substrate repair + limited functional smoke only if exploration substrate opens。

C6: D-RBF / FastKAN
  substrate repair + active-center / width / compact bump health。

C7: D-WAV
  low-budget substrate monitor。
```

## 4.2 Mechanism family 维度

v16.1 至少同时验证六类机制。它们不是 token search，而是互相不同的假设。

```text
M1: Productive pulse + recovery dynamics
  假设：短期坏 debt 可以通过长期训练动力学偿还。

M2: Split-consensus signal-subspace metric
  假设：跨 train split 一致的 signal subspace 才能产生可转移 update。

M3: PopRisk / drift-diffusion SNR preconditioner
  假设：coherent population signal 要靠 per-example gradient drift-vs-diffusion 识别。

M4: Function-space proximal / split-transfer solver
  假设：先在 function/output space 求小位移，再映射回参数，比直接参数 pulse 更合理。

M5: Optimizer-aware aligned FU
  假设：FU 失败来自与普通梯度、二阶矩、momentum dynamics 冲突。

M6: Carrier-specific reparameterized FU
  假设：source/hazard 不可分不是 FU 错，而是 carrier 坐标错。
```

每个 active carrier 至少运行 M1-M3。D-CHE 和 MLP 必须运行 M1-M5。LQ 若 reanchor 通过，运行 M1-M4。D-FOU/D-RBF/D-WAV 未过 substrate gate 时只运行 substrate + M1 smoke，不写 official functional。

---

# 5. Functional mechanism definitions

## 5.1 M1：Productive Pulse + Recovery Dynamics

目标：判断 bad update 是 unrecovered damage 还是 productive plasticity。

基本形式：

$$
\theta_{t+1}=\theta_t+u_{FU,t}
$$

之后不用同一个 FU 连续推，而是进入 recovery：

$$
\theta_{t+h}=\operatorname{Recovery}_{h}(\theta_{t+1}).
$$

预注册 recovery mechanisms：

```text
R0: ordinary AdamW recovery
R1: LR cooldown recovery
R2: cosine restart recovery
R3: momentum damping recovery
R4: second-moment adaptation recovery
R5: decoupled weight decay recovery
R6: role-wise / degree-wise decay recovery
R7: EMA / Lookahead / SWA-like consolidation
R8: feature recentering recovery
R9: late consolidation recovery
```

记录 horizon：

```text
H = 1, 5, 20, 50, 100, 400, 800, 1600
```

M1 不能因为 H=400 fail 阻断 H=800/H=1600。

---

## 5.2 M2：Split-Consensus Signal-Subspace Metric

目标：判断多个 train micro-splits 中一致的方向是否比单 batch update 更有泛化意义。

把当前 train batch 拆成 $K$ 个 split：

$$
B_1, B_2, \dots, B_K.
$$

计算每个 split 的 train gradient：

$$
g_j=\nabla_\theta \mathcal L_{B_j}(\theta).
$$

用二阶矩或 role-wise metric whiten：

$$
h_j=M_t^{-1/2}g_j.
$$

构造 consensus operator：

$$
C_{split}=\frac{1}{K(K-1)}\sum_{a\ne b}h_a h_b^T.
$$

构造 split-noise：

$$
N_{split}=\frac{1}{K}\sum_j(h_j-\bar h)(h_j-\bar h)^T.
$$

signal operator：

$$
A_{split}=C_{split}-\lambda_nN_{split}.
$$

更新不是找 action，而是让普通 gradient 只在 signal subspace / metric 中走：

$$
u_t=-M_t^{-1/2}P_{A}M_t^{-1/2}g_t.
$$

需要记录：

```text
split_count
signal_to_noise_ratio
effective_rank_A_split
top_eigenvalue_A_split
eigengap_A_split
negative_eigen_fraction
projection_retention_adam_grad
rolewise_consensus
degreewise_consensus
subspace_stability_across_steps
subspace_stability_across_seeds
```

---

## 5.3 M3：PopRisk / Drift-Diffusion SNR Preconditioner

目标：从 per-example gradient 中找 coherent signal，而不是追单步 geometry score。

对参数组或 role $r$：

$$
\mu_r=\frac{1}{b}\sum_{i=1}^b g_{i,r},
$$

$$
\sigma_r^2=\frac{1}{b}\sum_i(g_{i,r}-\mu_r)^2.
$$

定义 drift-diffusion score：

$$
SNR_r=\frac{\mu_r^2}{\sigma_r^2/(b-1)+\epsilon}.
$$

FU 不是另一个方向，而是对普通 gradient 的 preconditioner：

$$
\Delta\theta_r=-\eta \cdot p(SNR_r)\cdot \frac{m_{t,r}}{\sqrt{v_{t,r}}+\epsilon}.
$$

预注册 variants：

```text
M3a-HardSNRGate
M3b-SoftSNRGate
M3c-EMA3RoleNormSNR
M3d-SNRPlusSecondMoment
M3e-SNRPlusRecoveryPulse
```

注意：SNR 只使用 current train-stream per-example gradients，不使用 validation/test/future/audit metric。

---

## 5.4 M4：Function-Space Proximal / Split-Transfer Solver

目标：不是先造参数 update，而是在 function/output space 解一个小的 proximal problem。

候选子空间 $U$：

```text
U0: Adam subspace
U1: per-example gradient low-rank subspace
U2: D-CHE degree-role subspace
U3: LQ frame subspace, only if reanchor passed
U4: MLP hidden-channel subspace
U5: random-sketch output-Jacobian subspace
```

在 train split $B_1/B_2$ 上解：

$$
\alpha^*
=
\arg\min_\alpha
\mathcal L_{B_1}(f_\theta+J_{B_1}U\alpha)
+
\lambda_2\mathcal L_{B_2}(f_\theta+J_{B_2}U\alpha)
+
\rho\|\alpha\|_2^2
+
\tau\|U\alpha\|_{M_t}^2.
$$

固定 alpha candidates：

```text
0, 0.025, 0.05, 0.10, 0.20
```

不允许 adaptive search / controller。

---

## 5.5 M5：Optimizer-Aware Aligned FU

目标：验证 FU 是否因为和普通 gradient / momentum / second moment 冲突而失败。

给定 FU proposal：

$$
u_t=\Delta\theta_{FU}.
$$

普通下降方向：

$$
d_t=-\frac{m_t}{\sqrt{v_t}+\epsilon}.
$$

alignment：

$$
a_i=u_{t,i}d_{t,i}.
$$

variants：

```text
M5a-HardCautiousFU: a_i <= 0 则该坐标不更新。
M5b-SoftCautiousFU: a_i <= 0 只衰减，不清零。
M5c-RoleCautiousFU: 按 role / degree / group 汇总 alignment。
M5d-MGUPStyleReweight: 根据 momentum-gradient alignment 动态重权。
M5e-SophiaDiagLite: role-wise curvature scaling + clipping。
```

这条线必须配 `AdamW / CautiousAdamW / MGUP / random matched` controls，防止把 generic optimizer gain 写成 FU success。

---

## 5.6 M6：Carrier-Specific Reparameterized FU

目标：如果 source/hazard 在原坐标共线，检查换 carrier/coordinate 是否能分开。

D-CHE candidates：

```text
K1-LowDegreeSignalHighDegreeResidual
K2-ReadoutBasisDecoupled
K3-OrthogonalDegreeBank
K4-OutputJacobianCarrier
K5-DualBankSignalReservoir
```

LQ candidates：

```text
L1-HistoricalLQReplay
L2-LQReanchorCurrentProtocol
L3-LQSnapshotLateAttach
L4-LQSignalFramePulse
L5-LQMLPAnalogBridge
```

Non-D-CHE candidates 先做 substrate repair，不直接 official FU proof。

Carrier decoupling readback：

```text
mean_hazard_overlap
source_retention_after_null
source_vs_best_control_mean
bad_event_fraction
control_equivalent_fraction
```

---

# 6. v16.1 Carrier × Mechanism execution matrix

## 6.1 D-CHE matrix

D-CHE 必须执行：

```text
D-CHE-M1: G7R pulse + recovery family R0..R9
D-CHE-M2: split-consensus signal-subspace metric
D-CHE-M3: PopRisk / drift-diffusion SNR preconditioner
D-CHE-M4: function-space proximal solver on D-CHE degree-role / Jacobian subspace
D-CHE-M5: optimizer-aware aligned FU
D-CHE-M6: carrier reparameterization K1..K5 readback + top-pass update
```

D-CHE controls：

```text
NoOpMatchedOverhead
RandomMatchedPulseSameRecovery
SameNormRandomPulse
SameActiveFractionRandomPulse
AdamWExtraStepsMatchedTime
RecoveryOnlyNoPulse
DecayOnly, if decay involved
MLPAnalog
GenericOptimizerControl
```

---

## 6.2 MLP matrix

MLP 不再是 control；它是 active line。

MLP 必须执行：

```text
MLP-M1: functional pulse + recovery R0..R9
MLP-M2: split-consensus hidden-channel metric
MLP-M3: PopRisk / drift-diffusion SNR preconditioner
MLP-M4: function-space proximal on hidden low-rank / output-Jacobian sketch
MLP-M5: optimizer-aware aligned FU
MLP-M6: hidden-channel carrier factorization readback
```

MLP 的成功解释：

```text
MLP positive + controls fail:
  generic functional dynamics insight。

MLP positive + D-CHE fail:
  FU mechanism 可能成立，KAN carrier 不会承载。

MLP fail + D-CHE fail:
  当前 FU family 可能 no-go。

D-CHE positive + MLP weaker:
  才能讨论 KAN-specific advantage。
```

---

## 6.3 LQ matrix

LQ 必须先 reanchor。

LQ reanchor：

```text
LQ-C0: Historical LQ reference replay
LQ-C1: Current protocol clean reanchor
LQ-C2: Snapshot late attach anchor
LQ-C3: LQ linec/no-regression monitor
LQ-C4: LQ efficiency / AUC / near-pass hardening
```

若 reanchor gate 打开，执行：

```text
LQ-M1: LQ functional pulse + recovery
LQ-M2: LQ split-consensus signal frame
LQ-M3: LQ PopRisk/SNR preconditioner
LQ-M4: LQ snapshot late-attach functional dynamics
```

若 reanchor gate 不开：

```text
functional rows fail-closed deferred；
必须输出 protocol drift / row-level gate fragility / historical-current flip table。
```

---

## 6.4 Rational matrix

Rational 只做 monitor 与 sanity，不重启 reset route。

执行：

```text
RAT-M0: Rational AdamW monitor
RAT-M1: Rational PopRisk/SNR monitor
RAT-M2: Rational split-consensus readback
RAT-M3: Rational no-reset no-regression
RAT-Ctrl: Random matched / AdamW controls
```

禁止：

```text
reset route；
optimizer-state transport；
controller；
action bank；
seed/dataset branch。
```

---

## 6.5 D-FOU / D-RBF / D-WAV matrix

这些 family 当前只进入 substrate repair + limited smoke。

D-FOU：

```text
FOU-S1: low-frequency identity residual
FOU-S2: bandwise SNR warmup
FOU-S3: phase-stable band mix
FOU-S4: high-frequency quarantine
FOU-S5: no-materialize lifetime / optimizer-state colocation
```

D-RBF / FastKAN：

```text
RBF-S1: active center occupancy
RBF-S2: width condition guard
RBF-S3: compact bump no-dense materialization
RBF-S4: Gaussian local K4 task-health
RBF-S5: center/readout split-consensus monitor
```

D-WAV：

```text
WAV-S1: triangular support
WAV-S2: scale occupancy
WAV-S3: support overlap damping
WAV-S4: local-tail coverage audit
```

如果某 family 达到 exploration substrate gate：

```text
family_dataset_seed_pass_count >= 6/9
```

则允许 low-budget M1/M3 functional smoke；未达到则不允许 official FU proof。

---

# 7. 指标记录：每个 row 必须统一落盘

## 7.1 基本字段

```text
run_id
carrier
mechanism_family
method
control_type
dataset
seed
horizon
train_steps
recovery_type
pulse_type
subspace_type
metric_type
direction_provenance
uses_validation_for_direction
uses_test_for_direction
uses_future_for_direction
uses_query_for_direction
uses_linec_for_direction
uses_tail_audit_for_direction
uses_dataset_name_branch
uses_seed_specific_scale
```

## 7.2 Dynamic debt fields

```text
source_vs_best_control_h1/h5/h20/h50/h100/h400/h800/h1600
source_retention_h1/h5/h20/h50/h100/h400/h800/h1600
tail_debt_peak
tail_debt_final_horizon
tail_recovery_rate_horizon
LineC_debt_peak
LineC_debt_final_horizon
LineC_recovery_rate_horizon
calibration_debt_peak
calibration_debt_final_horizon
calibration_recovery_rate_horizon
AUC_debt_peak
AUC_debt_final_horizon
AUCtime_ratio_horizon
DGS_horizon
bad_event_fraction_horizon
```

## 7.3 Split / transfer fields

```text
B1_loss_delta
B2_loss_delta
split_agreement
split_disagreement
micro_horizon_loss_integral_h1/h2/h4
train_motion_probe_motion_R2
CouplingR2_delta
NoiseSignalLeak_delta
RealSignalReservoirRatio_delta
signal_channel_energy
reservoir_energy
noise_leakage_proxy
```

## 7.4 Optimizer / recovery fields

```text
adam_update_norm
fu_update_norm
recovery_update_norm
recovery_to_pulse_norm_ratio
recovery_cosine_to_pulse
recovery_cosine_to_adam
momentum_norm
second_moment_mean
second_moment_p90
decay_update_norm
rolewise_decay_norm
ema_distance
lookahead_sync_count
swa_distance
lr_cooldown_factor
restart_count
```

## 7.5 Efficiency fields

```text
step_time_ratio_H
memory_ratio_H
forward_time_ms
backward_time_ms
update_time_ms
recovery_overhead_ms
fu_overhead_ms
samples_per_second
cpu_offload_used
fake_proxy_used
```

---

# 8. 成功标准

## 8.1 S1：matrix coverage completed

```text
Line R complete;
D-CHE M1..M6 complete;
MLP M1..M6 complete;
LQ reanchor complete; if opened, LQ M1..M4 complete;
Rational monitor complete;
D-FOU/RBF/WAV substrate complete;
Line M controls complete;
Line Z route + exhaustion certificate complete.
```

## 8.2 S2：weak productive dynamics

至少一个 carrier × mechanism 满足：

```text
source_vs_best_control_h800 >= 0.005
source_retention_h800 >= 0.40
tail_recovery_rate_h800 >= 0.40 or LineC_recovery_rate_h800 >= 0.40
AUCtime_ratio_h800 <= 1.10
matched controls fail
```

这是 weak exploration，不允许 promotion。

## 8.3 S3：productive debt recovery

至少一个 carrier × mechanism 满足：

```text
source_vs_best_control_h800 >= 0.005
source_retention_h800 >= 0.50
tail_recovery_rate_h800 >= 0.60
LineC_recovery_rate_h800 >= 0.60
AUCtime_ratio_h800 <= 1.05
random pulse + same recovery fails
recovery-only fails
NoOp overhead fails
```

## 8.4 S4：real-transfer exploration

```text
real_lite_pass_count >= 6/9
source_vs_best_control_mean >= 0.005
AUCtime_median <= 1.05
tail debt recovered
LineC debt recovered
controls fail
```

## 8.5 S5：official success，不降低

```text
real_dataset_seed_pass_count = 9/9
source_vs_best_control >= 0.005
AUCtime_ratio <= 1.0
CEp99_delta <= 0.05
NLL_delta <= 0.02
ECE_delta <= 0.02
LineC_pass = 1
step_time_ratio <= 1.25
memory_ratio <= 1.25
controls fail
forbidden audit pass
no-action-search audit pass
code review pass
promotion_allowed = 1
```

---

# 9. 4 GPU 并行执行计划

当前服务器有 4 张 GPU。v16.1 必须显式充分利用，不允许单卡串行跑完整计划。

## 9.1 Round 0：统一 smoke / provenance / registry audit

```text
GPU0: D-CHE method registry + smoke
GPU1: MLP method registry + smoke
GPU2: LQ/Rational registry + smoke
GPU3: D-FOU/RBF/WAV substrate registry + smoke
```

Round 0 产出：

```text
method_surface_manifest.csv
carrier_registry_manifest.csv
forbidden_information_audit.csv
no_action_search_audit.csv
gpu_assignment_manifest.csv
```

## 9.2 Round 1：short-horizon mechanism matrix

```text
GPU0:
  D-CHE M1/M2/M3, H=1/5/20/50/100, with controls.

GPU1:
  MLP M1/M2/M3, H=1/5/20/50/100, with controls.

GPU2:
  LQ reanchor C0..C4 + Rational M0..M3.
  If LQ reanchor opens during this round, enqueue LQ M1/M2 short-horizon.

GPU3:
  D-FOU / D-RBF / D-WAV substrate repair rows.
```

Round 1 结束必须立刻合并：

```text
short_horizon_matrix_summary.csv
positive_looking_rows.csv
control_required_queue.csv
```

## 9.3 Round 2：long-horizon extension and alternate mechanism families

```text
GPU0:
  D-CHE M4/M5/M6 + top D-CHE M1/M2/M3 to H=400/800.

GPU1:
  MLP M4/M5/M6 + top MLP M1/M2/M3 to H=400/800.

GPU2:
  LQ functional if reanchor gate opened; otherwise deeper LQ reanchor / late-attach audit.
  Rational monitor continuation.

GPU3:
  all-basis continued substrate + any family reaching >=6/9 gets limited M1/M3 smoke.
```

Round 2 结束必须合并：

```text
h800_matrix_summary.csv
carrier_by_mechanism_heatmap.csv
source_retention_vs_debt_recovery.csv
```

## 9.4 Round 3：h1600 consolidation only for top candidates

Top candidates 选择不能用 validation/test/future，也不能用 audit metric 作为方向；只用 already-logged train-stream source/debt readback 与 controls coverage。

```text
GPU0:
  top D-CHE candidate h1600.

GPU1:
  top MLP candidate h1600.

GPU2:
  top LQ/Rational candidate if any h1600 or reanchor continuation.

GPU3:
  top all-basis substrate / limited functional smoke continuation.
```

h1600 不要求每个 method 都跑，但要求：

```text
top-2 source-retaining candidates per active carrier；
all positive-looking MLP/KAN paired attribution rows；
all candidate controls needed for promotion/no-go decision。
```

## 9.5 GPU utilization audit

必须记录：

```text
gpu_id
assigned_lines
assigned_methods
start_time
end_time
wall_clock_used
rows_planned
rows_executed
rows_failed
rows_deferred
deferred_reason
utilization_percent_estimate
```

如果任一 GPU 长时间空闲且仍有 runnable queue，Codex 必须写明原因；否则视为 execution-contract fail。

---

# 10. Visualizations

v16.1 必须生成以下图，不允许只写 CSV。

## 10.1 Carrier × mechanism dashboard

```text
carrier_mechanism_gate_heatmap.svg
carrier_mechanism_source_heatmap_h800.svg
carrier_mechanism_debt_recovery_heatmap_h800.svg
carrier_mechanism_control_equivalence_heatmap.svg
```

## 10.2 Dynamic recovery dashboard

```text
source_retention_curve_by_carrier.svg
tail_debt_curve_by_carrier.svg
LineC_debt_curve_by_carrier.svg
calibration_debt_curve_by_carrier.svg
AUCtime_curve_by_carrier.svg
recovery_rate_by_mechanism.svg
```

## 10.3 MLP vs KAN attribution dashboard

```text
D-CHE_vs_MLP_source_retention.svg
D-CHE_vs_MLP_tail_recovery.svg
D-CHE_vs_MLP_LineC_recovery.svg
KAN_specific_delta_heatmap.svg
generic_vs_kan_specific_route_dashboard.svg
```

## 10.4 LQ / all-basis dashboards

```text
LQ_reanchor_nearpass_heatmap.svg
LQ_protocol_drift_heatmap.svg
basis_family_pass_count_heatmap.svg
basis_step_memory_pareto.svg
basis_LineC_task_health.svg
```

## 10.5 GPU utilization dashboard

```text
gpu_assignment_timeline.svg
gpu_rows_executed_by_round.svg
gpu_idle_reason_table.svg
```

---

# 11. Failure handling：不能浅尝辄止，也不能乱搜

## 11.1 Promotion fail-closed

No S5, no promotion. No exceptions.

## 11.2 Exploration matrix-open

这些情况不能 hard stop：

```text
D-CHE M1 fail；
MLP M1 fail；
LQ reanchor fail；
Rational monitor fail；
D-FOU/RBF/WAV < 6/9；
M1 pulse + recovery fail；
M2 split-consensus fail；
M3 PopRisk/SNR fail；
M4 proximal fail；
M5 optimizer-aware fail；
M6 carrier reparameterization fail；
MLP/generic controls positive；
LineC/tail/AUC immediate fail；
h400 fail；
h800 fail；
overhead high；
random pulse explains result。
```

它们只能进入：

```text
failure taxonomy；
fallback ladder；
exhaustion certificate；
next hypothesis queue。
```

## 11.3 Hard stop only for violations

只有以下情况允许 hard stop：

```text
required_artifact_missing_count > 0；
forbidden_information_violation_count > 0；
no_action_search_violation_count > 0；
direction uses validation/test/future/query；
direction uses LineC/CEp99/NLL/ECE/AUCtime/Brier；
dataset-name branch；
seed-specific scale；
action token / controller / action bank / reset route；
fake / proxy / CPU offload。
```

## 11.4 Budget exhaustion certificate

预算耗尽不能成为偷停理由。只有写出下列字段，才允许 stop：

```text
budget_kind: gpu_time / wall_clock / rows / seeds / horizon / fallback_depth / family_count
planned_budget
consumed_budget
mandatory_executed
fallback_executed
deferred_items
deferred_reason
whether_deferred_items_affect_route
next_priority_queue
final_stop_allowed
```

If `mandatory_executed=0` or `fallback_executed=0`, final stop is invalid unless forbidden/hard-stop violation occurred.

---

# 12. Codex implementation and review requirements

Codex 必须在实验结果复盘文件中写明：

```text
1. 本轮新增/修改了哪些文件；
2. 每个 functional mechanism 的核心实现函数在哪些文件/函数；
3. 每个 carrier 如何构造；
4. direction provenance 如何保证 train-stream-only；
5. controls 如何构造；
6. recovery horizon 如何执行；
7. 4 GPU 如何分配；
8. 哪些 rows 是真实训练，哪些是 readback/audit；
9. 哪些项因为预算 deferred；
10. deferred 是否影响 final route；
11. 重要实现和现存代码的思路说明。
```

必须给出 code review packet，至少包括：

```text
runner source;
method registry;
carrier factory;
functional update implementations;
control implementations;
recovery implementations;
finalizer;
route decision logic;
required artifact manifest;
forbidden/no-action/provenance audits;
gpu assignment manifest。
```

核心代码必须特别审查：

```text
1. FU direction 是否真的不读取 validation/test/future/query。
2. LineC/tail/AUC/calibration 是否只 readback，不参与 direction。
3. MLP active line 是否真的执行，而不是只作为 control。
4. LQ reanchor gate 是否正确 fail-closed。
5. all-basis substrate rows 是否没有被写成 official FU success。
6. four-GPU shards 是否完整合并，没有遗漏 rows。
7. h400 fail 是否没有阻断 h800/h1600。
8. positive-looking rows 是否都有 controls。
```

---

# 13. Final route decision rules

## Case A：D-CHE positive, MLP negative

```text
KAN-specific functional advantage possible.
Proceed to S4/S5 confirmation only if controls fail and debt recovered.
```

## Case B：MLP positive, D-CHE negative

```text
Generic functional dynamics insight possible.
Do not claim KAN-specific.
Investigate carrier limitation / KAN coordinate issue.
```

## Case C：D-CHE and MLP both positive

```text
Generic training-dynamics success.
KAN-specific delta required only for KAN-specific claim.
```

## Case D：all carriers fail all mechanisms

```text
route = R16_1-FunctionalMechanismMatrixNoGo
Stop current functional update family.
Next must be theory-level reset or substrate/base-level redesign.
```

## Case E：LQ reanchor opens but functional fails

```text
LQ carrier viable but current FU mechanisms fail.
Continue LQ carrier analysis, not promotion.
```

## Case F：D-FOU/RBF/WAV substrate opens

```text
Allow next-version official FU proof on that family.
Do not retroactively claim success in v16.1.
```

---

# 14. 最终判断

v16.1 的核心不是再写一个 functional update，而是把问题改成一次性并行裁决：

$$
\boxed{
\text{在 D-CHE、MLP、LQ、Rational、D-FOU/RBF/WAV 上，}
\text{到底哪一种 functional 机制能产生长期 productive dynamics？}
}
$$

如果只有 MLP positive，说明机制可能是 generic；如果只有 KAN positive，才可能是 KAN-specific；如果所有 carrier 和所有机制都失败，就必须停止当前 train-stream functional-update family，而不是继续 token/action/controller/reset。

这一版计划的执行重点是：

```text
1. 多 carrier；
2. 多 functional mechanism；
3. long-horizon debt recovery；
4. strict controls；
5. 4 GPU 并行；
6. route/no-go/exhaustion 全闭环。
```

一句话：

$$
\boxed{
\text{不要再用一条线等结果；}
\text{用四张 GPU 同时裁决多套 functional dynamics 假设。}
}
$$
