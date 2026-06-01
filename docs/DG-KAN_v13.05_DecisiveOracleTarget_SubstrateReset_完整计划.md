# DG-KAN v13.5：Decisive Oracle Target / Legal Observability / Substrate Reset 完整计划

> 版本：v13.5 execution plan  
> 基于：v13.4 `OperatorLevelBasisChannelFunctional` 真实结果复盘  
> 公式格式：Typora 友好，只使用 `$...$` 与 `$$...$$`  
> 核心原则：不再继续 BM/BN 参数度量小修，不再继续 O7/O8 可达 target 小修，不把 S1C / O/P actuation pass 写成 functional success。下一步必须用 oracle upper-bound 与 legal observability 分解，判定当前 functional update 失败到底是 target 错、substrate 不够，还是合法可观测性不足。

---

# 0. 项目总目标与当前进展

## 0.1 项目总目标

DG-KAN 的总目标仍然是：

$$
\boxed{
\text{构建 label-free strict FC-PureKAN substrate/base，}
\text{并通过 loss-interface-generic functional update，}
\text{得到比普通反向传播更好的模型与训练几何。}
}
$$

这里的成功不是单个 diagnostic 指标变好，而是必须同时满足：

```text
1. label-free：不得使用 label-informed initialization，例如 trainprobe / y_for_stats。
2. strict FC-PureKAN：不引入普通 MLP stem/head/LN 作为隐藏支撑。
3. efficiency：forward / backward / step / memory 与 MLP 可比。
4. expression：表达力不打折。
5. task trajectory：AUC-step / AUC-time 不输或接近 MLP。
6. geometry：train-probe coupling、signal/reservoir/noise、tail、calibration 不坏，最好改善。
7. functional update：必须打过 NoOp / RandomMatchedNorm / AdamWParallel / SNR-only / MLP analog controls。
8. loss-interface-generic：functional direction 可通过统一 loss interface 接收 output cotangent，但不能 hardcode CE-specific formula；CEp99 / NLL / ECE / LineC hard target 只允许作为 audit / gate，不能生成方向。
```

最终要证明：

$$
\boxed{
\text{PureKAN substrate/base + functional update}
>
\text{same substrate/base + ordinary backprop / AdamW controls}
}
$$

## 0.2 当前真实进展

v13.4 是一个重要的战略分界点。它没有成功，但它把问题切得比之前更准。

v13.3 之前的问题是：我们在参数空间里做 diagonal / low-rank / block metric update，结果 P3/P4 全部失败。v13.4 改成 operator-level basis-channel route：

```text
loss cotangent -> desired basis-channel movement DeltaZ
-> parameter projection J_theta->Z
-> true basis-parameter writeback
-> synthetic task-family future proof
```

v13.4 的真实结果是：

```text
route = R3-SyntheticProofFail
minimum_success = S1C-ChannelControllableSubstrate
official_success_reached = 0
promotion_allowed = 0
operator_gate_pass_count = 54 / 56
parameter_projection_pass_count = 54 / 56
synthetic_rows = 14
synthetic_task_success_count = 0
synthetic_5of7_pass = 0
substrate_s1c_count = 1
nonrat_s1c_count = 0
required_artifact_missing_count = 0
provenance_violation_count = 0
```

这说明：

```text
1. S1C 已经第一次成立：至少一个 substrate 能执行 reachable basis-channel DeltaZ。
2. O/P actuation 已经不是主要 blocker：operator gate 与 parameter projection gate 大量通过。
3. Synthetic future proof 完全没过：0/14 synthetic rows pass，0/7 task family pass。
4. source_vs_best 最大只有约 0.00231，低于 0.005 gate。
5. Non-RAT 仍然没有 S1C。
6. 当前 operator target formulation 已经进入 no-go 边界，不能继续小修 O7/O8。
```

所以 v13.5 的核心不是继续问：

```text
我们能不能执行 DeltaZ？
```

而是问：

$$
\boxed{
\text{是否存在一个可产生 future advantage 的 basis-channel target，}
\text{并且这个 target 是否能被合法 precommit 地估计？}
}
$$

---

# 1. 各条线当前进展百分比

这些百分比不是 artifact 官方字段，而是基于 gate 通过情况、机制清晰度、代码审计完整度和距离 official success 的综合估计。

| 线 | v13.3 后估计 | v13.4 后估计 | 变化 | 判断 |
|---|---:|---:|---:|---|
| Line R：代码 / provenance / finalizer 审计 | 99% | **99%** | 0 | 工程闭包强，manifest / provenance clean；不是当前 blocker。 |
| Historical FHQ / B320-current 工程能力 | 85% frozen | **85% frozen** | 0 | 历史强，但 label-informed init 已禁用，不能 official。 |
| Label-free FHQ / A-DYN monitor | 20% | **18%** | -2 | 多轮无 near-anchor，继续低预算 monitor。 |
| Line C 几何审计 | 88% | **88%** | 0 | 审计可靠，但不作为 direction source。 |
| Rational substrate S1 | 85% | **85%** | 0 | 仍是唯一稳定 S1 substrate family。 |
| Rational S1C channel controllability | 0%-5% | **55%** | +50 | v13.4 首次建立 S1C；但只在 Rational / current channel formulation。 |
| Non-RAT substrate / S1C | 30%-38% | **25%-35%** | -5 | Non-RAT S1C = 0；仍不能进入 operator-level functional proof。 |
| Full basis-param writeback | 60% | **75%** | +15 | 真参数 writeback + actual DeltaZ measurement 可稳定执行。 |
| Operator-level basis-channel solve | 0%-5% | **45%** | +40 | O/P 大量通过，但只表示可执行，不表示有未来收益。 |
| Synthetic future proof | 5%-10% | **5%** | -5 | 0/14 pass；source_vs_best 不足。 |
| MLP analog channel control | 18% | **15%** | -3 | 无 positive；generic MLP functional 继续不乐观。 |
| Basis-specific functional official | 0%-5% | **0%-5%** | 0 | S5 仍为 0。 |
| Overall next-gen MLP claim | 38%-43% | **39%-44%** | +1 | S1C 是真实进展，但 synthetic proof 全失败，所以整体只小幅上调。 |

核心解读：**v13.4 的能力进展不是模型变好，而是证明“执行想要的 basis-channel 位移”这件事终于可行；但这个位移没有产生可验证的未来训练收益。**

---

# 2. v13.4 独立分析

## 2.1 v13.4 不是失败到没有价值

v13.4 做对了几件以前没有做到的事：

```text
1. 不再使用 readout-feature proxy / frozen feature table。
2. 定义了 live basis-channel Z。
3. 计算真实 J_theta->Z sketch。
4. 真实写回 basis parameters，然后测 actual DeltaZ，再回滚。
5. 发现 hidden activation channel 太低秩，改成 live rational basis channel 后 rank blocker 明显缓解。
6. O7 actuation-aware reachable target 把 DeltaZ 投入 J_theta->Z 可执行子空间，首次得到 S1C。
7. O8 reachable-logit fallback 也被测试，没有偷偷停在 O7。
8. scale / horizon / optimizer-state transport fallback 都执行了。
```

这说明 v13.4 不是 Codex 一下停了，也不是 artifact-only no-go。它真正推进了实现层级。

## 2.2 但 v13.4 不是 functional success

它没有产生任何可 promotion 的 synthetic proof：

```text
synthetic_rows = 14
synthetic_task_success_count = 0
synthetic_5of7_pass = 0
source_vs_best max = 0.00231058 < 0.005
```

最接近的 X4 row 有很大 CouplingR2 delta、NoiseSignalLeak 下降和 Reservoir release，但 source_vs_best 仍然只有约 0.00231，不够打过 best control / task baseline。

这说明：

$$
\boxed{
\text{当前 O7/O8 operator target 能移动几何，}
\text{但不是能带来未来训练收益的 value target。}
}
$$

## 2.3 blocker 已从 actuation 转为 value alignment

v13.3 的问题是：parameter metric route 过不了。v13.4 的问题更具体：

```text
S1C 可执行；
O/P actuation pass；
future source advantage 不足。
```

所以现在不能再把失败解释成：

```text
参数写不进去；
projection fidelity 不够；
需要回到 BN/BM metric；
需要继续调 O7/O8 scale；
需要直接 real short-run。
```

v13.4 已经证明这些小修没有把 source_vs_best 推过 0.005。

真正的问题是：

$$
\boxed{
\text{我们不知道什么 DeltaZ 是有价值的；}
\text{能执行 DeltaZ 不等于 DeltaZ 有价值。}
}
$$

---

# 3. 当前根本问题

## 3.1 我们缺的不是 actuator，而是 target

过去的问题是：

```text
functional update 能不能动？
```

现在答案变成：

```text
能动，至少在 Rational S1C 上能动。
```

但新问题是：

```text
该动什么？
```

O7 把 loss-interface cotangent 投到可执行 basis-channel 子空间，这是一个自然 target，但实验表明它不够。O8 把 output cotangent 经过 reachable logit 子空间回投，也不够。

因此，我们需要进行 **target upper-bound decomposition**，而不是继续发明 O9/O10。

## 3.2 必须拆开三个问题

当前失败可能来自三种完全不同的原因：

```text
A. Target problem:
   存在有价值的 DeltaZ，但 O7/O8 没找到。

B. Substrate problem:
   即使给 oracle DeltaZ，当前 Rational S1C substrate 也无法带来未来训练收益。

C. Legal observability problem:
   oracle DeltaZ 可以成功，但任何 precommit / loss-interface-generic feature 都看不到它。
```

这三种原因需要完全不同的下一步。如果不拆开，我们会继续陷入低价值搜索。

## 3.3 关键判定矩阵

v13.5 必须用一个 decisive matrix 来定路线：

| 实验 | 是否允许 promotion | 目的 |
|---|---:|---|
| O-current：O7/O8 reachable target | 否，除非过 synthetic 5/7 | 当前合法 target baseline |
| O-oracle-future-DeltaZ | 否 | 测 substrate 是否存在可利用 target upper bound |
| O-oracle-task-family-specific | 否 | 测是不是 task-family-specific target 才有用 |
| O-precommit-proxy | 可进入 promotion path | 测合法 feature 能否近似 oracle |
| O-random-reachable-control | 否，control | 测 oracle / legal target 是否只是 norm / subspace artifact |
| MLP-hidden-oracle | 否，control | 测机制是否 KAN-specific |

结论规则：

```text
1. oracle 也失败：当前 substrate/operator formulation no-go，回 substrate/base architecture。
2. oracle 成功，precommit 失败：target legal observability no-go，不能继续 O token 小修。
3. oracle 成功，precommit 成功：进入 official synthetic / real short-run。
4. MLP oracle 也成功：机制不是 KAN-specific，必须重新定义 claim。
```

---

# 4. v13.5 核心假设

## H1：当前失败主要是 target 错，不是 actuation 错

v13.4 的 54/56 O/P pass 支持这一点。我们需要验证是否存在 oracle DeltaZ 能带来 synthetic future gain。

如果没有 oracle DeltaZ 能成功，那么继续 functional target 搜索没有意义。

## H2：Rational S1C 可能只提供可执行通道，不提供有价值通道

S1C 只说明：

$$
J_{\theta\to Z}\Delta\theta \approx \Delta Z.
$$

它不说明：

$$
\Delta Z \Rightarrow \text{better future training}.
$$

v13.5 要测 oracle target upper bound。如果 oracle 都不行，Rational S1C 不是 functional substrate，只是 actuation substrate。

## H3：Non-RAT S1C 缺失可能限制泛化

目前 Non-RAT S1C 为 0。Rational-only signal 可能导致 X family coverage 太窄。v13.5 必须并行做至少一个 Non-RAT S1C vertical slice，而不是继续只看 Rational。

## H4：single-event functional update 可能不是正确形式

如果 single-step oracle DeltaZ 失败，但 two-event / sequential oracle 成功，说明 functional update 应该是 multi-stage coordinate transport，而不是单次 update。

如果 two-event oracle 也失败，则当前 substrate/operator route no-go。

---

# 5. v13.5 实验总结构

```text
Line R：Implementation / provenance / code review。
Line S：Substrate/S1C status and Non-RAT S1C rescue。
Line O：Decisive Oracle Target Upper Bound。
Line V：Legal Precommit Target Visibility。
Line P：Parameter projection and actuation fidelity audit。
Line X：Synthetic proof and target decomposition。
Line Q：Sequential / two-event oracle diagnostic。
Line M：MLP analog oracle and legal target control。
Line D：Classic no-BSpline substrate continuation。
Line Z：Finalizer / no-go boundary / next decision。
```

---

# 6. Line R：Implementation / provenance / code review

## 6.1 目标

确保 v13.5 的 oracle / precommit / projection / synthetic proof 语义干净。尤其要避免：

```text
1. oracle target 混入 promotion；
2. precommit target 偷看 future / validation / query batch；
3. CEp99 / NLL / ECE / LineC hard target 被用作 direction source；
4. readout-feature proxy 回流；
5. frozen feature table 伪装 basis-channel target；
6. Non-RAT S1C 使用不真实 kernel / fake projection。
```

## 6.2 必须落盘

```text
v135_code_review_manifest.csv
v135_target_provenance_manifest.csv
v135_oracle_target_audit.csv
v135_precommit_feature_audit.csv
v135_projection_writeback_trace.csv
v135_forbidden_information_audit.csv
v135_required_artifact_manifest.csv
```

## 6.3 硬门

如果任一 promotion candidate 出现：

```text
uses_future_for_direction = 1
uses_validation_for_direction = 1
uses_test_for_direction = 1
uses_query_batch_for_direction = 1
uses_linec_hard_target_for_direction = 1
uses_cep99_nll_ece_for_direction = 1
readout_feature_proxy_only = 1
full_basis_param_update_rows = 0
```

则 route：

```text
R0-ProvenanceViolation
```

---

# 7. Line S：Substrate / S1C status and Non-RAT rescue

## 7.1 目标

v13.4 已经有 Rational S1C，但 Non-RAT S1C 为 0。v13.5 要把 substrate 分为三类：

```text
S1: efficient substrate。
S1C: channel-controllable substrate。
S2: healthy base。
```

## 7.2 必须记录

```text
family
candidate
workspace_raw_ratio
workspace_incremental_ratio
step_ratio
substrate_s1_pass
basis_channel_rank_ratio
delta_z_effective_rank
operator_gate_pass
projection_gate_pass
actuation_error
actuation_cosine
substrate_s1c_pass
```

## 7.3 Non-RAT S1C vertical slice

至少选择两个 Non-RAT family：

```text
Fourier low-K exact kernel
Chebyshev K3/K4 exact kernel
```

只做 S1C feasibility，不直接 functional promotion。

### 7.3.1 Fourier S1C repair

尝试：

```text
FOU-S1C1: sincos channel Z with band-normalized DeltaZ。
FOU-S1C2: frequency-band block projection。
FOU-S1C3: low-frequency-only reachable target。
```

### 7.3.2 Chebyshev S1C repair

尝试：

```text
CHE-S1C1: degree-channel Z with degree-energy normalized DeltaZ。
CHE-S1C2: low-degree-only projection。
CHE-S1C3: recurrence-stable channel projection。
```

## 7.4 Gate

S1C pass：

$$
actuation\_error \le 0.35,
$$

$$
cos(\Delta Z_{actual},\Delta Z^\star) \ge 0.50,
$$

$$
predicted\_logit\_drift \le 0.35,
$$

$$
actual\_logit\_drift \le 0.35.
$$

Non-RAT 若仍无 S1C：

```text
NonRAT_S1C = 0
functional proof remains Rational-bound
```

---

# 8. Line O：Decisive Oracle Target Upper Bound

## 8.1 目标

判断：当前 S1C substrate 是否存在“有价值的 DeltaZ”。

这一步允许使用 oracle future / label 信息，但只作为 **diagnostic upper bound**，绝不允许 promotion。

## 8.2 Oracle target 类型

### O-OR1：Future-training DeltaZ oracle

在 synthetic task 上，先运行普通 TaskOnlyAdamW 未来 $k$ 步，记录 basis-channel 变化：

$$
\Delta Z_{oracle}=Z_{t+k}^{TaskOnly}-Z_t.
$$

然后把 $\Delta Z_{oracle}$ 投影到当前可执行子空间，并执行一次 functional event，比较未来 probe。

### O-OR2：Best-control residual oracle

定义：

$$
\Delta Z_{oracle}=\Delta Z_{best\ future}-\Delta Z_{best\ control}.
$$

目的是判断是否存在超越 best control 的 residual basis-channel target。

### O-OR3：LineC-release oracle

只作为 audit upper bound：使用 LineC hard release 后验构造 target，检查如果直接向 release 方向移动，是否能带来 synthetic gain。

不能 promotion。

### O-OR4：Task-family-specific oracle

对每个 X family 单独拟合 oracle target，判断 target 是否 family-specific。

## 8.3 必须记录

```text
oracle_type
task_family
seed
uses_future_for_direction
uses_label_for_direction
promotion_allowed = 0
oracle_delta_z_norm
oracle_projected_delta_z_norm
oracle_projection_residual
oracle_actuation_error
oracle_actuation_cosine
source_vs_best
CouplingR2_delta
NoiseSignalLeak_delta
Reservoir_delta
CEp99_delta
NLL_delta
ECE_delta
synthetic_success
```

## 8.4 Decisive gates

Oracle diagnostic pass：

$$
source\_vs\_best \ge 0.005
$$

on at least:

```text
5 / 7 synthetic families
```

or:

```text
>= 4 / 7 with leave-family-out robustness and no tail disaster.
```

If oracle fails:

```text
route = R4-OracleTargetNoUpperBound
recommendation = stop functional on current substrate; return to substrate/base architecture
```

If oracle passes:

```text
continue to Line V legal precommit visibility.
```

---

# 9. Line V：Legal Precommit Target Visibility

## 9.1 目标

如果 oracle 成功，判断是否能用合法 precommit feature 预测 oracle target 或 target class。

## 9.2 Allowed features

只允许：

```text
current logits
unlabeled activations
basis telemetry
basis-channel covariance
J_theta->Z sketch
J_Z->Y sketch
optimizer state excluding label-specific gradients
random cotangent response
train-stream unlabeled perturbation response
```

禁止：

```text
future result
validation/test/query batch
LineC hard target
CEp99/NLL/ECE hard target
permuted-label CE
hard release label
```

## 9.3 Outputs

预测：

```text
oracle_target_family
oracle_target_direction_lowrank
oracle_success_probability
expected_source_vs_best
expected_tail_risk
```

## 9.4 Gate

Heldout split：

$$
AUC_{oracle\_success} \ge 0.75,
$$

$$
precision@k \ge 0.30,
$$

$$
recall@k \ge 0.30.
$$

如果 oracle pass 但 visibility fail：

```text
route = R5-LegalObservabilityNoGo
recommendation = current functional target not deployable under legal constraints
```

---

# 10. Line P：Projection and actuation audit

## 10.1 目标

继续保证所有 target 都真实写回 basis params，且 actual DeltaZ 与 target 匹配。

## 10.2 Solvers

```text
P3-LowRankWoodburyProjection
P4-ConjugateGradientJtJProjection
P7-ReachableSubspaceProjection
P8-TrustRegionReachableProjection
```

## 10.3 Metrics

```text
actuation_error
actuation_cosine
actual_delta_z_norm
predicted_logit_drift
actual_logit_drift
parameter_update_norm
update_norm_ratio
writeback_success
rollback_error
```

## 10.4 Gate

$$
actuation\_error \le 0.35,
$$

$$
cos \ge 0.50,
$$

$$
actual\_logit\_drift \le 0.35.
$$

---

# 11. Line X：Synthetic task-family proof

## 11.1 目标

只有 Line S/O/P 或 Line V/P 通过后，才运行 synthetic proof。

## 11.2 Tasks

```text
X1 pairwise product
X2 rotated quadratic
X3 random quadratic
X4 low/high frequency mixture
X5 local bump / tail cluster
X6 two-manifold class split
X7 previous working family
```

## 11.3 Controls

```text
TaskOnlyAdamW
NoOpMatchedOverhead
RandomMatchedNorm
AdamWParallelDirection
SNR-only
ReachableRandomDeltaZ
OracleUnprojectedDiagnostic
```

## 11.4 Gate

Synthetic success：

$$
source\_vs\_best \ge 0.005,
$$

$$
CouplingR2\_delta \ge 0.02,
$$

$$
NoiseSignalLeak\_delta \le 0,
$$

$$
Reservoir\_delta \le 0,
$$

$$
CEp99\_delta \le 0.05,
$$

$$
NLL\_delta \le 0.02,
$$

$$
ECE\_delta \le 0.02.
$$

Synthetic promotion path requires:

```text
>= 5 / 7 task family pass
```

Real short-run remains closed unless synthetic gate passes.

---

# 12. Line Q：Sequential / two-event oracle diagnostic

## 12.1 目标

If single-event oracle fails, test whether the failure is because functional update must be sequential.

## 12.2 Methods

```text
Q1: Oracle geometry-guard then task target
Q2: Oracle task target then geometry-guard
Q3: small DeltaZ repeated over 2 events
Q4: coordinate transport + short refresh
```

## 12.3 Gate

If single oracle fails but sequential oracle passes:

```text
functional update should be low-frequency maintenance sequence, not one-shot event.
```

If sequential oracle fails:

```text
current substrate/operator route no-go.
```

---

# 13. Line M：MLP analog oracle and legal target control

## 13.1 目标

判断 operator-level channel target 是否 KAN-specific。

MLP analog channel：hidden activation $H$。

## 13.2 Experiments

```text
M-OR1: MLP hidden-channel oracle DeltaH
M-OR2: MLP hidden-channel precommit proxy
M-OR3: MLP hidden-channel random reachable control
```

## 13.3 Interpretation

```text
If MLP oracle and precommit both pass:
  mechanism is generic representation-channel functional update.

If MLP oracle fails but KAN oracle passes:
  KAN explicit basis-channel coordinate may be essential.

If both fail:
  operator-level functional target no-go.
```

---

# 14. Line D：Classic no-BSpline substrate continuation

## 14.1 Goal

Continue active no-BSpline basis families, but not as token search.

Active families:

```text
Rational
Chebyshev
Fourier
RBF/FastKAN
Wavelet
```

B-spline remains frozen.

## 14.2 Rational

Rational stays primary S1C family. v13.5 tests oracle target decomposition on Rational first.

## 14.3 Non-RAT

Non-RAT must target S1C, not direct P3.

```text
Chebyshev: degree-channel controllability.
Fourier: frequency-band controllability.
RBF: center/width occupancy controllability.
Wavelet: scale/support controllability.
```

Each family must report:

```text
WorkspaceOnly
EfficientSubstrate
ChannelControllableSubstrate
HealthyBase
NoGoCurrentImplementation
```

---

# 15. 必须记录的统一指标

## 15.1 Progress / route

```text
route
minimum_success
official_success_reached
promotion_allowed
final_stop_allowed
hard_compute_budget_exhausted
fallback_all_executed
```

## 15.2 Substrate / S1C

```text
family
candidate
substrate_s1_pass
substrate_s1c_pass
basis_channel_rank_ratio
delta_z_effective_rank
operator_gate_pass
projection_gate_pass
actuation_error
actuation_cosine
```

## 15.3 Oracle

```text
oracle_type
uses_future_for_direction
uses_label_for_direction
promotion_allowed
oracle_projection_residual
oracle_actuation_error
source_vs_best
synthetic_success
```

## 15.4 Precommit visibility

```text
feature_family
AUC_success
precision_at_k
recall_at_k
leave_family_out_AUC
precommit_allowed
forbidden_feature_count
```

## 15.5 Functional audit

```text
source_vs_noop
source_vs_random
source_vs_adamw_parallel
source_vs_snr
source_vs_best
CouplingR2_delta
NoiseSignalLeak_delta
RealSignalReservoirRatio_delta
CEp99_delta
NLL_delta
ECE_delta
```

---

# 16. 必须生成的可视化

```text
fig_progress_by_line.svg
fig_s1c_status_by_family.svg
fig_actuation_error_vs_source_vs_best.svg
fig_delta_z_target_vs_actual.svg
fig_current_vs_oracle_target_synthetic.svg
fig_oracle_projection_residual_by_task.svg
fig_precommit_visibility_auc_precision_recall.svg
fig_synthetic_family_pass_heatmap.svg
fig_nonrat_s1c_status.svg
fig_mlp_vs_kan_oracle.svg
fig_failure_taxonomy.svg
```

---

# 17. Stop / Go 规则

## 17.1 禁止继续的方向

如果 v13.5 仍然没有 oracle upper-bound，则禁止继续：

```text
1. O7/O8/O9 token 小修；
2. BM/BN metric 回退；
3. response dictionary vN；
4. real short-run；
5. LineC hard target direction；
6. CEp99/NLL/ECE direction；
7. readout-feature proxy；
8. frozen feature-table transport。
```

## 17.2 必须继续的方向

如果 oracle upper-bound pass：

```text
1. legal precommit visibility；
2. constrained precommit target construction；
3. synthetic 5/7；
4. then real short-run。
```

如果 oracle upper-bound fail：

```text
return to substrate/base architecture.
```

If oracle pass but legal visibility fail:

```text
record legal observability no-go and stop functional under current constraints.
```

---

# 18. Final route definitions

```text
S1: EfficientSubstrate。
S1C: ChannelControllableSubstrate。
S2: OracleTargetUpperBoundPass。
S3: LegalPrecommitTargetVisible。
S4: Synthetic5of7FunctionalProof。
S5: RealDataOfficialFunctionalSuccess。

R0: Implementation/provenance fail。
R1: NoS1C。
R2: CurrentTargetNoValue。
R3: OracleTargetPassButLegalVisibilityFail。
R4: OracleTargetNoUpperBound。
R5: MLPAnalogExplainsMechanism。
R6: NonRATSubstrateMissingButRationalOnlyMechanism。
R7: ReadyForRealShortRun。
```

---

# 19. 结论

v13.4 证明了一件很重要的事：

$$
\boxed{
\text{我们已经能在 Rational S1C 上执行 reachable basis-channel DeltaZ，}
\text{但当前 DeltaZ 没有带来 future training advantage。}
}
$$

因此 v13.5 不再继续小修 operator token，而是做一个 decisive decomposition：

```text
1. oracle DeltaZ 有没有 upper-bound？
2. 如果有，legal precommit features 能不能看见它？
3. 如果没有，当前 substrate/operator formulation 是否应该 no-go？
4. Non-RAT 是否能建立 S1C，避免 Rational-only 结论？
5. MLP analog 是否解释这个机制？
```

这才是下一轮真正能决定路线的实验。
